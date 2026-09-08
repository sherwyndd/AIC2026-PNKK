import json
import logging
from typing import Any, Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Clipboard"])

# In-memory store for clipboard state
_clipboard_store: Dict[str, Any] = {
    "kisRanking": [],
    "qaRanking": [],
    "trakeRanking": [],
    "savedImages": [],
    "queryHistory": [],
    "noteText": "",
}

_active_websockets: Set[WebSocket] = set()


class ClipboardUpdateRequest(BaseModel):
    data: Dict[str, Any]
    action: str = "Cập nhật Clipboard"


@router.get("/api/clipboard")
async def get_clipboard():
    return {
        "status": "ok",
        "data": _clipboard_store,
        "onlineUsers": max(1, len(_active_websockets)),
    }


@router.post("/api/clipboard")
async def update_clipboard(payload: ClipboardUpdateRequest):
    global _clipboard_store
    _clipboard_store = payload.data
    # Broadcast to all connected WebSockets
    msg = json.dumps({
        "type": "SYNC",
        "data": _clipboard_store,
        "action": payload.action,
        "onlineUsers": max(1, len(_active_websockets)),
    })
    disconnected = set()
    for ws in _active_websockets:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    _active_websockets.difference_update(disconnected)
    return {"status": "ok"}


@router.websocket("/ws/clipboard")
async def websocket_clipboard_endpoint(websocket: WebSocket):
    await websocket.accept()
    _active_websockets.add(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "INIT",
            "data": _clipboard_store,
            "onlineUsers": len(_active_websockets),
        }))
        # Broadcast presence update
        presence_msg = json.dumps({
            "type": "PRESENCE",
            "onlineUsers": len(_active_websockets),
        })
        for ws in list(_active_websockets):
            if ws != websocket:
                try:
                    await ws.send_text(presence_msg)
                except Exception:
                    pass

        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                if data.get("type") == "PING":
                    continue
                elif data.get("type") == "UPDATE":
                    if "data" in data and isinstance(data["data"], dict):
                        _clipboard_store = data["data"]
                    sync_msg = json.dumps({
                        "type": "SYNC",
                        "data": _clipboard_store,
                        "action": data.get("action", "UPDATE"),
                        "onlineUsers": len(_active_websockets),
                    })
                    for ws in list(_active_websockets):
                        if ws != websocket:
                            try:
                                await ws.send_text(sync_msg)
                            except Exception:
                                _active_websockets.discard(ws)
            except Exception as e:
                logger.debug("WS message handling error: %s", e)
    except WebSocketDisconnect:
        pass
    finally:
        _active_websockets.discard(websocket)
        presence_msg = json.dumps({
            "type": "PRESENCE",
            "onlineUsers": max(1, len(_active_websockets)),
        })
        for ws in list(_active_websockets):
            try:
                await ws.send_text(presence_msg)
            except Exception:
                _active_websockets.discard(ws)
