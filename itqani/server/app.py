"""
Thread 5 — FastAPI server with WebSocket broadcast.

Endpoints
─────────
GET  /           → téléprompteur HTML page
GET  /ws         → WebSocket for translation tokens
GET  /health     → JSON health check
"""

import asyncio
import logging
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Itqani Khutba Translator")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Set of active WebSocket connections
_connections: Set[WebSocket] = set()
_connections_lock = asyncio.Lock()


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "connections": len(_connections)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with _connections_lock:
        _connections.add(websocket)
    logger.info("WebSocket connected (%d total)", len(_connections))
    try:
        while True:
            # Keep connection alive; we only push from server side
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _connections_lock:
            _connections.discard(websocket)
        logger.info("WebSocket disconnected (%d remaining)", len(_connections))


async def broadcast(message: str) -> None:
    """Send a raw JSON string to all connected WebSocket clients."""
    if not _connections:
        return
    async with _connections_lock:
        targets = list(_connections)
    dead = set()
    for ws in targets:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    if dead:
        async with _connections_lock:
            _connections.difference_update(dead)


def make_sync_broadcast(loop: asyncio.AbstractEventLoop):
    """
    Return a thread-safe synchronous broadcast function that schedules
    the async broadcast coroutine on the given event loop.
    """
    def _sync_broadcast(message: str) -> None:
        asyncio.run_coroutine_threadsafe(broadcast(message), loop)

    return _sync_broadcast
