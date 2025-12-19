# Face Recognition Debug Tool

Web-based debug tool for Grove Vision AI Module V2 face recognition system.

## Features

- **Real-time Preview**: View video stream with detected face bounding boxes
- **Face Recognition**: Automatic name recognition from enrolled database
- **Face Enrollment**: 5-second capture mode to enroll new faces
- **Database Management**: View and delete enrolled faces

## Architecture

```
┌──────────────────┐      UART (921600)       ┌─────────────────────┐
│  Grove Vision    │  ─────────────────────▶  │   Python Backend    │
│  AI Module V2    │   JSON + Base64 Image    │   (FastAPI)         │
│                  │   + Embedding Data       │                     │
└──────────────────┘                          │  - Serial Parser    │
                                              │  - Face Database    │
                                              │  - WebSocket Server │
                                              └─────────┬───────────┘
                                                        │ WebSocket
                                                        ▼
                                              ┌─────────────────────┐
                                              │   Web Frontend      │
                                              │   (HTML/JS/CSS)     │
                                              │                     │
                                              │  - Video Canvas     │
                                              │  - Face Overlays    │
                                              │  - Enroll UI        │
                                              └─────────────────────┘
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Grove Vision AI Module V2 with face recognition firmware
- USB connection to the module

## Installation & Usage

### Using uv (Recommended)

```bash
cd tools/face_recognition_debug

# Install dependencies and run (one command)
uv run python backend/main.py

# Or sync first, then run
uv sync
uv run python backend/main.py
```

### Using pip

```bash
cd tools/face_recognition_debug/backend
pip install -r requirements.txt
python main.py
```

## Quick Start

1. **Flash the firmware** with face recognition support:
   ```bash
   # From project root
   ./build_and_flash.sh
   ```

2. **Start the server**:
   ```bash
   cd tools/face_recognition_debug
   uv run python backend/main.py
   ```

3. **Open the web interface**:
   - Navigate to http://localhost:4242
   - Select the serial port (usually `/dev/tty.usbmodem*` on macOS)
   - Click "Connect"

4. **Enroll a face**:
   - Enter a name in the "Face Enrollment" section
   - Click "Start Enrollment (5s)"
   - Look at the camera for 5 seconds
   - The system will collect multiple samples and calculate average embedding

5. **Recognition**:
   - Enrolled faces will be automatically recognized
   - Name and similarity score displayed in real-time

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/ports` | GET | List serial ports |
| `/api/connect` | POST | Connect to serial port |
| `/api/disconnect` | POST | Disconnect |
| `/api/status` | GET | Connection status |
| `/api/database` | GET | List enrolled faces |
| `/api/database/{name}` | DELETE | Delete a face |
| `/api/enroll` | POST | Start enrollment |
| `/api/enroll/status` | GET | Enrollment progress |
| `/api/enroll/cancel` | POST | Cancel enrollment |
| `/ws` | WebSocket | Real-time updates |

## Data Format

The firmware sends JSON data over UART:

```json
{
  "type": 1,
  "name": "FACE_RESULT",
  "code": 0,
  "data": {
    "image": "<base64_encoded_jpeg>",
    "resolution": [640, 480],
    "faces": [{
      "bbox": [x, y, w, h],
      "confidence": 0.95,
      "landmarks": [[x1,y1], [x2,y2], ...],
      "embedding": [0.1, 0.2, ...]
    }]
  }
}
```

## Database

Face embeddings are stored in SQLite (`faces.db`):

```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    embedding BLOB,      -- 512 floats
    sample_count INTEGER,
    created_at TIMESTAMP
);
```

## Recognition Algorithm

- **Similarity**: Cosine similarity between embeddings
- **Threshold**: 0.5 (configurable)
- **Higher similarity = better match**

## Troubleshooting

### No video stream
- Check serial port connection
- Verify firmware is running (check serial console output)
- Try refreshing serial ports list

### Low FPS
- JPEG encoding is resource-intensive
- Consider reducing resolution in firmware

### Recognition accuracy
- Ensure good lighting during enrollment
- Keep face centered and still
- Enroll with multiple angles if needed
