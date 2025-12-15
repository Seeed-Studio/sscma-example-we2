"""
Serial Reader Module
Parses JSON data from Grove Vision AI Module V2 UART output
"""

import serial
import json
import threading
import time
import base64
import struct
from typing import Callable, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FaceResult:
    image_base64: str
    resolution: tuple
    bbox: list
    confidence: float
    landmarks: list
    embedding: np.ndarray


class SerialReader:
    def __init__(self, port: str, baudrate: int = 921600):
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.callbacks: list[Callable[[FaceResult], None]] = []
        self.raw_callbacks: list[Callable[[dict], None]] = []
        self.buffer = ""

    def add_callback(self, callback: Callable[[FaceResult], None]):
        """Add callback for parsed face results"""
        self.callbacks.append(callback)

    def add_raw_callback(self, callback: Callable[[dict], None]):
        """Add callback for raw JSON data"""
        self.raw_callbacks.append(callback)

    def start(self) -> bool:
        """Start serial reading thread"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"Serial reader started on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to open serial port: {e}")
            return False

    def stop(self):
        """Stop serial reading thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial:
            self.serial.close()
            self.serial = None
        print("Serial reader stopped")

    def _read_loop(self):
        """Main reading loop"""
        bytes_received = 0
        last_log_time = time.time()

        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    data = self.serial.read(self.serial.in_waiting)
                    bytes_received += len(data)
                    self.buffer += data.decode('utf-8', errors='ignore')
                    self._process_buffer()

                    # Log every 2 seconds
                    if time.time() - last_log_time > 2:
                        print(f"[Serial] Received {bytes_received} bytes, buffer size: {len(self.buffer)}")
                        if len(self.buffer) > 0:
                            print(f"[Serial] Buffer preview: {self.buffer[:200]}...")
                        last_log_time = time.time()
                        bytes_received = 0
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)

    def _process_buffer(self):
        """Process buffer and extract JSON messages"""
        while True:
            # Find JSON start - support both \r{ and just { at line start
            start_idx = -1
            for pattern in ['\r{', '\n{']:
                idx = self.buffer.find(pattern)
                if idx != -1 and (start_idx == -1 or idx < start_idx):
                    start_idx = idx

            if start_idx == -1:
                # Try finding standalone { at the beginning
                if self.buffer.startswith('{'):
                    start_idx = -1  # Will be adjusted to 0 below
                else:
                    # No JSON start found, keep last 100 chars in case partial
                    if len(self.buffer) > 100:
                        self.buffer = self.buffer[-100:]
                    break

            # Adjust start_idx to point to '{'
            if start_idx == -1:
                start_idx = 0
            else:
                start_idx += 1  # Skip \r or \n

            # Find JSON end - look for "}}" followed by newline or end
            # Try multiple ending patterns
            end_idx = -1
            end_len = 0
            for end_pattern in ['}}\r\n', '}}\n', '}}\r']:
                idx = self.buffer.find(end_pattern, start_idx)
                if idx != -1 and (end_idx == -1 or idx < end_idx):
                    end_idx = idx
                    end_len = len(end_pattern)

            if end_idx == -1:
                # No complete JSON end found
                # But don't let buffer grow too large (limit to 100KB)
                if len(self.buffer) > 100000:
                    print(f"[Serial] Warning: Buffer too large ({len(self.buffer)}), clearing...")
                    self.buffer = ""
                break

            # Extract JSON string
            json_str = self.buffer[start_idx:end_idx + 2]  # Include }}
            self.buffer = self.buffer[end_idx + end_len:]  # Remove processed part

            # Parse JSON
            self._parse_json(json_str)

    def _parse_json(self, json_str: str):
        """Parse JSON and notify callbacks"""
        import time as _time
        parse_start = _time.perf_counter()

        try:
            data = json.loads(json_str)
            json_parse_time = (_time.perf_counter() - parse_start) * 1000
            # Show last 50 chars to debug JSON ending
            json_tail = json_str[-50:] if len(json_str) > 50 else json_str
            print(f"[Serial] Parsed JSON: type={data.get('type')}, name={data.get('name')}, json_len={len(json_str)}, parse_time={json_parse_time:.1f}ms, tail=...{json_tail}")

            # Notify raw callbacks
            for callback in self.raw_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Raw callback error: {e}")

            # Check if it's a face result
            if data.get('name') == 'FACE_RESULT' and data.get('code') == 0:
                face_data = data.get('data', {})
                faces = face_data.get('faces', [])
                print(f"[Serial] FACE_RESULT: image_len={len(face_data.get('image', ''))}, faces={len(faces)}")
                if faces:
                    print(f"[Serial] Face data: bbox={faces[0].get('bbox')}, conf={faces[0].get('confidence')}, has_emb={bool(faces[0].get('embedding'))}")

                # Extract image
                image_base64 = face_data.get('image', '')
                resolution = tuple(face_data.get('resolution', [0, 0]))

                # Extract faces
                faces = face_data.get('faces', [])

                if faces:
                    # Process each detected face
                    for face in faces:
                        bbox = face.get('bbox', [0, 0, 0, 0])
                        confidence = face.get('confidence', 0.0)
                        landmarks = face.get('landmarks', [])

                        # Convert embedding to numpy array
                        # Support both JSON array and Base64 formats
                        embedding_format = face.get('embedding_format', 'array')
                        embedding_data = face.get('embedding', [])
                        # Get embedding dimension from JSON (defaults to 512 for backward compatibility)
                        embedding_dim = face.get('embedding_dim', 512)

                        if embedding_format == 'float32_base64' and isinstance(embedding_data, str) and embedding_data:
                            # Decode Base64 binary format (N floats = N*4 bytes)
                            try:
                                binary_data = base64.b64decode(embedding_data)
                                embedding = np.frombuffer(binary_data, dtype=np.float32)
                            except Exception as e:
                                print(f"[Serial] Base64 decode error: {e}")
                                embedding = np.zeros(embedding_dim, dtype=np.float32)
                        elif isinstance(embedding_data, list) and embedding_data:
                            # JSON array format - use actual length
                            embedding = np.array(embedding_data, dtype=np.float32)
                        else:
                            embedding = np.zeros(embedding_dim, dtype=np.float32)

                        result = FaceResult(
                            image_base64=image_base64,
                            resolution=resolution,
                            bbox=bbox,
                            confidence=confidence,
                            landmarks=landmarks,
                            embedding=embedding
                        )

                        # Notify callbacks
                        print(f"[Serial] Notifying {len(self.callbacks)} callbacks (with face)")
                        for callback in self.callbacks:
                            try:
                                callback(result)
                            except Exception as e:
                                print(f"Callback error: {e}")
                else:
                    # No face detected, still send image frame
                    # Get embedding dimension from face_data if available (defaults to 512 for backward compatibility)
                    embedding_dim = face_data.get('embedding_dim', 512)
                    result = FaceResult(
                        image_base64=image_base64,
                        resolution=resolution,
                        bbox=[0, 0, 0, 0],
                        confidence=0.0,
                        landmarks=[],
                        embedding=np.zeros(embedding_dim, dtype=np.float32)
                    )

                    # Notify callbacks with empty face
                    print(f"[Serial] Notifying {len(self.callbacks)} callbacks (no face, image only)")
                    for callback in self.callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"Callback error: {e}")

        except json.JSONDecodeError as e:
            # Partial or invalid JSON, ignore
            pass
        except Exception as e:
            print(f"Parse error: {e}")


def list_serial_ports() -> list:
    """List available serial ports"""
    import serial.tools.list_ports
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append({
            "device": port.device,
            "description": port.description,
            "hwid": port.hwid
        })
    return ports
