"""
Clipboard Manager: Thread-safe, atomic-persisted, real-time shared clipboard manager for FastAPI.
Provides in-memory caching, atomic JSON file persistence on disk, auto-backup,
and WebSocket connection management with broadcast capabilities.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket

logger = logging.getLogger("clipboard_manager")

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
BACKUP_DIR = CACHE_DIR / "backups"
CLIPBOARD_FILE = CACHE_DIR / "shared_clipboard.json"

DEFAULT_STATE: Dict[str, Any] = {
    "version": 1,
    "activeQueryId": "query-1",
    "kisRanking": [],
    "qaRanking": [],
    "trakeRanking": [],
    "savedImages": [],
    "noteText": "",
    "queryHistory": [],
    "updatedAt": "",
    "lastAction": "INIT",
    "lastModifiedBy": "system",
}


class ClipboardManager:
    def __init__(self):
        self._state: Dict[str, Any] = dict(DEFAULT_STATE)
        self._lock = asyncio.Lock()
        self._active_connections: Set[WebSocket] = set()
        self._ensure_dirs()
        self._load_from_disk()

    def _ensure_dirs(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    def _load_from_disk(self):
        """Load state from shared_clipboard.json if exists and valid, otherwise fallback."""
        if CLIPBOARD_FILE.exists():
            try:
                with open(CLIPBOARD_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Merge with default state to ensure all keys exist
                        merged = dict(DEFAULT_STATE)
                        merged.update(data)
                        self._state = merged
                        logger.info(
                            f"[ClipboardManager] Loaded shared clipboard (v{merged.get('version', 1)}): "
                            f"KIS={len(merged.get('kisRanking', []))}, "
                            f"QA={len(merged.get('qaRanking', []))}, "
                            f"TRAKE={len(merged.get('trakeRanking', []))}, "
                            f"Note={len(merged.get('savedImages', []))}"
                        )
                        return
            except Exception as e:
                logger.error(f"[ClipboardManager] Failed to read {CLIPBOARD_FILE}: {e}. Creating backup of corrupted file.")
                try:
                    corrupted_file = BACKUP_DIR / f"corrupted_{int(time.time())}.json"
                    shutil.copy(CLIPBOARD_FILE, corrupted_file)
                except Exception:
                    pass

        self._state = dict(DEFAULT_STATE)
        self._state["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save_to_disk_sync()

    def _save_to_disk_sync(self):
        """Atomic write to disk: write to temp file then replace."""
        tmp_file = CACHE_DIR / f"shared_clipboard.json.tmp.{os.getpid()}"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, CLIPBOARD_FILE)
        except Exception as e:
            logger.error(f"[ClipboardManager] Atomic file write failed: {e}")
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def create_backup(self, reason: str = "auto"):
        """Create a timestamped snapshot backup in cache/backups/."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_file = BACKUP_DIR / f"clipboard_{reason}_{timestamp}.json"
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            logger.info(f"[ClipboardManager] Backup created: {backup_file.name}")
        except Exception as e:
            logger.error(f"[ClipboardManager] Backup creation failed: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current state."""
        return dict(self._state)

    def get_active_count(self) -> int:
        return len(self._active_connections)

    async def update_state(
        self,
        new_data: Dict[str, Any],
        action: str = "UPDATE",
        sender: str = "anonymous",
    ) -> Dict[str, Any]:
        """Thread-safe state update and atomic persistence."""
        async with self._lock:
            # Check if this is a major clear action to trigger auto-backup
            if action.startswith("CLEAR") or not new_data.get("kisRanking"):
                if len(self._state.get("kisRanking", [])) > 5:
                    self.create_backup("pre_clear")

            # Update keys
            current_version = self._state.get("version", 0) + 1
            updated = dict(self._state)
            
            for key in ["kisRanking", "qaRanking", "trakeRanking", "savedImages", "noteText", "queryHistory", "activeQueryId"]:
                if key in new_data:
                    updated[key] = new_data[key]

            updated["version"] = current_version
            updated["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            updated["lastAction"] = action
            updated["lastModifiedBy"] = sender

            self._state = updated
            
            # Non-blocking file save in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_to_disk_sync)

            return dict(self._state)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active_connections.add(websocket)
        logger.info(f"[ClipboardManager] Client connected. Total online: {len(self._active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self._active_connections.discard(websocket)
        logger.info(f"[ClipboardManager] Client disconnected. Total online: {len(self._active_connections)}")

    async def broadcast(self, message: Dict[str, Any], exclude: Optional[WebSocket] = None) -> None:
        """Broadcast a message to all connected clients except optional excluded one."""
        if not self._active_connections:
            return

        dead_connections: List[WebSocket] = []
        tasks = []

        for ws in self._active_connections:
            if ws != exclude:
                tasks.append((ws, ws.send_json(message)))

        if not tasks:
            return

        results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                dead_connections.append(tasks[i][0])

        for dead_ws in dead_connections:
            self._active_connections.discard(dead_ws)


# Global singleton instance
clipboard_manager = ClipboardManager()
