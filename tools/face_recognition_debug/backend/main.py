"""
Face Recognition Debug Tool - Backend Server

FastAPI application providing:
- WebSocket for real-time video + face results
- REST API for face enrollment and database management
- Serial port communication with Grove Vision AI Module V2
"""

import asyncio
import json
import os
from typing import Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

from serial_reader import SerialReader, FaceResult, list_serial_ports
from face_database import FaceDatabase, EnrollmentSession


# Global state
serial_reader: Optional[SerialReader] = None
face_db: Optional[FaceDatabase] = None
active_websockets: Set[WebSocket] = set()
enrollment_session: Optional[EnrollmentSession] = None
enrollment_lock = asyncio.Lock()
main_event_loop: Optional[asyncio.AbstractEventLoop] = None  # Store main event loop reference


class EnrollRequest(BaseModel):
    name: str
    duration: float = 5.0


class ConnectRequest(BaseModel):
    port: str
    baudrate: int = 921600


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global face_db, main_event_loop

    # Startup
    print("Starting Face Recognition Debug Server...")

    # Store main event loop reference for cross-thread async calls
    main_event_loop = asyncio.get_running_loop()

    face_db = FaceDatabase("faces.db")
    print(f"Database initialized with {len(face_db.get_all_faces())} faces")

    yield

    # Shutdown
    global serial_reader
    if serial_reader:
        serial_reader.stop()
    print("Server shutdown complete")


app = FastAPI(
    title="Face Recognition Debug Tool",
    description="Real-time face recognition debug and enrollment",
    lifespan=lifespan
)


# Serve static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def root():
    """Serve main page"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Face Recognition Debug Tool API"}


@app.get("/api/ports")
async def get_ports():
    """List available serial ports"""
    return {"ports": list_serial_ports()}


@app.post("/api/connect")
async def connect_serial(request: ConnectRequest):
    """Connect to serial port"""
    global serial_reader

    # Stop existing connection
    if serial_reader:
        serial_reader.stop()
        serial_reader = None

    # Create new connection
    serial_reader = SerialReader(request.port, request.baudrate)
    serial_reader.add_raw_callback(on_raw_face_data)  # Multi-face debug mode
    serial_reader.add_callback(on_face_result)

    if serial_reader.start():
        return {"status": "connected", "port": request.port}
    else:
        serial_reader = None
        raise HTTPException(status_code=500, detail="Failed to connect to serial port")


@app.post("/api/disconnect")
async def disconnect_serial():
    """Disconnect from serial port"""
    global serial_reader
    if serial_reader:
        serial_reader.stop()
        serial_reader = None
    return {"status": "disconnected"}


@app.get("/api/status")
async def get_status():
    """Get connection status"""
    return {
        "connected": serial_reader is not None and serial_reader.running,
        "port": serial_reader.port if serial_reader else None,
        "websockets": len(active_websockets),
        "faces_in_db": len(face_db.get_all_faces()) if face_db else 0
    }


@app.get("/api/database")
async def get_database():
    """Get all enrolled faces"""
    if not face_db:
        return {"faces": []}

    faces = face_db.get_all_faces()
    return {
        "faces": [
            {
                "id": f.id,
                "name": f.name,
                "sample_count": f.sample_count,
                "created_at": f.created_at
            }
            for f in faces
        ]
    }


@app.delete("/api/database/{name}")
async def delete_face(name: str):
    """Delete a face from database"""
    if not face_db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    if face_db.delete_face(name):
        await broadcast_message({
            "type": "database_update",
            "action": "delete",
            "name": name
        })
        return {"status": "deleted", "name": name}
    else:
        raise HTTPException(status_code=404, detail="Face not found")


@app.post("/api/enroll")
async def start_enrollment(request: EnrollRequest):
    """Start face enrollment session"""
    global enrollment_session

    async with enrollment_lock:
        if enrollment_session and enrollment_session.active and not enrollment_session.is_expired():
            raise HTTPException(status_code=400, detail="Enrollment already in progress")

        # Check if name already exists
        if face_db:
            existing = [f for f in face_db.get_all_faces() if f.name == request.name]
            if existing:
                raise HTTPException(status_code=400, detail=f"Name '{request.name}' already exists")

        enrollment_session = EnrollmentSession(
            name=request.name,
            duration=request.duration
        )

        # Start enrollment timer
        asyncio.create_task(enrollment_timer(request.duration))

        return {
            "status": "started",
            "name": request.name,
            "duration": request.duration
        }


@app.get("/api/enroll/status")
async def get_enrollment_status():
    """Get current enrollment status"""
    global enrollment_session

    if not enrollment_session:
        return {"active": False}

    return enrollment_session.get_progress()


@app.post("/api/enroll/cancel")
async def cancel_enrollment():
    """Cancel current enrollment"""
    global enrollment_session

    async with enrollment_lock:
        if enrollment_session:
            enrollment_session.active = False
            enrollment_session = None

    return {"status": "cancelled"}


async def enrollment_timer(duration: float):
    """Timer to finalize enrollment after duration"""
    global enrollment_session

    await asyncio.sleep(duration)

    async with enrollment_lock:
        if enrollment_session and enrollment_session.active:
            embedding = enrollment_session.finalize()

            if embedding is not None and face_db:
                # Save to database
                success = face_db.add_face(
                    enrollment_session.name,
                    embedding,
                    len(enrollment_session.embeddings)
                )

                await broadcast_message({
                    "type": "enrollment_complete",
                    "success": success,
                    "name": enrollment_session.name,
                    "samples": len(enrollment_session.embeddings)
                })

                if success:
                    await broadcast_message({
                        "type": "database_update",
                        "action": "add",
                        "name": enrollment_session.name
                    })
            else:
                await broadcast_message({
                    "type": "enrollment_complete",
                    "success": False,
                    "name": enrollment_session.name if enrollment_session else "Unknown",
                    "samples": len(enrollment_session.embeddings) if enrollment_session else 0,
                    "error": "Not enough samples"
                })

            enrollment_session = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    print(f"WebSocket connected. Total: {len(active_websockets)}")

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Handle client commands if needed
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        active_websockets.discard(websocket)
        print(f"WebSocket disconnected. Total: {len(active_websockets)}")


async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not active_websockets:
        return

    text = json.dumps(message)
    disconnected = set()

    for ws in active_websockets:
        try:
            await ws.send_text(text)
        except:
            disconnected.add(ws)

    # Remove disconnected clients
    for ws in disconnected:
        active_websockets.discard(ws)


# Counter to skip faces that were already broadcast via raw callback
_skip_face_count = 0


def on_raw_face_data(data: dict):
    """Raw callback for multi-face broadcasting (debug mode)"""
    global _skip_face_count

    if data.get('name') != 'FACE_RESULT' or data.get('code') != 0:
        return

    face_data = data.get('data', {})
    faces = face_data.get('faces', [])
    image_base64 = face_data.get('image', '')

    # Only handle multi-face case (debug mode) here
    # Single face or no face will be handled by on_face_result
    if len(faces) <= 1:
        return

    print(f"[Raw] Multi-face broadcast: {len(faces)} faces")

    # Build message with ALL faces
    message = {
        "type": "face_result",
        "image": image_base64,
        "resolution": face_data.get('resolution', [0, 0]),
        "faces": [
            {
                "bbox": f.get('bbox', [0, 0, 0, 0]),
                "confidence": f.get('confidence', 0.0),
                "landmarks": f.get('landmarks', []),
                "recognized_name": None,
                "similarity": 0.0
            }
            for f in faces
        ]
    }

    # Set skip count for subsequent on_face_result calls
    _skip_face_count = len(faces)

    # Broadcast to all WebSocket clients
    if main_event_loop is not None:
        asyncio.run_coroutine_threadsafe(
            broadcast_message(message),
            main_event_loop
        )


def on_face_result(result: FaceResult):
    """Callback when face result is received from serial port"""
    global enrollment_session, _skip_face_count

    # Skip if already broadcast by raw callback (multi-face mode)
    if _skip_face_count > 0:
        _skip_face_count -= 1
        return

    # Process enrollment if active
    if enrollment_session and enrollment_session.active and not enrollment_session.is_expired():
        enrollment_session.add_sample(result.embedding, result.confidence)

    # Recognize face
    recognized_name = None
    similarity = 0.0
    if face_db and result.embedding is not None:
        recognized_name, similarity = face_db.recognize(result.embedding)

    # Build message for WebSocket
    message = {
        "type": "face_result",
        "image": result.image_base64,
        "resolution": list(result.resolution),
        "faces": [{
            "bbox": result.bbox,
            "confidence": result.confidence,
            "landmarks": result.landmarks,
            "recognized_name": recognized_name,
            "similarity": similarity
        }]
    }

    # Add enrollment progress if active
    if enrollment_session and enrollment_session.active:
        message["enrollment"] = enrollment_session.get_progress()

    # Broadcast to all WebSocket clients (from serial thread to main async loop)
    if main_event_loop is not None:
        asyncio.run_coroutine_threadsafe(
            broadcast_message(message),
            main_event_loop
        )
    else:
        print("[Warning] Main event loop not available, skipping broadcast")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4242)


if __name__ == "__main__":
    main()
