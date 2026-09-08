"""
Pydantic schemas for DRES requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── DRES Low-level schemas ────────────────────────────────────────────────────

class DresLoginRequest(BaseModel):
    """Payload for POST /api/v2/login"""
    username: Optional[str] = None
    password: Optional[str] = None


class DresLoginResponse(BaseModel):
    """Response from DRES /api/v2/login or backend POST /api/dres/login"""
    sessionId: Optional[str] = Field(default=None, description="Session ID from DRES")
    status: bool = Field(default=True, description="Login status flag")
    message: Optional[str] = Field(default=None, description="Descriptive message")


class DresAnswerItem(BaseModel):
    """Single answer item in DRES KIS/QA/TraKE payload."""
    mediaItemName: str = Field(..., description="Video ID (e.g. L26_V320)")
    start: Optional[int] = Field(default=None, description="Start millisecond offset")
    end: Optional[int] = Field(default=None, description="End millisecond offset")
    text: Optional[str] = Field(default=None, description="Optional text for QA answers")


class DresAnswerSet(BaseModel):
    """Answer set grouping answers."""
    answers: List[DresAnswerItem] = Field(default_factory=list)


class DresSubmissionPayload(BaseModel):
    """Top-level body sent to DRES POST /api/v2/submit/{evaluationId}?session={sessionId}"""
    answerSets: List[DresAnswerSet] = Field(default_factory=list)


# ── Backend API Endpoints schemas ─────────────────────────────────────────────

class DresSubmitRequest(BaseModel):
    """
    Request model for POST /api/dres/submit.
    Supports both standard search submission and VQA submission.
    """
    video_id: str = Field(..., description="Video ID (mediaItemName), e.g. 'L26_V320'")
    timestamp: Optional[int] = Field(default=None, description="Timestamp in milliseconds, e.g. 145000")
    timestamp_ms: Optional[int] = Field(default=None, description="Timestamp in milliseconds (alias)")
    start: Optional[int] = Field(default=None, description="Optional start ms (defaults to timestamp)")
    end: Optional[int] = Field(default=None, description="Optional end ms (defaults to timestamp)")
    text: Optional[str] = Field(default=None, description="Optional text answer for QA")
    answer: Optional[str] = Field(default=None, description="VQA answer text, e.g. 'TIKTOK'")
    frames: Optional[List[Any]] = Field(default=None, description="List of frame indices for TRAKE sequence submission")
    frame_ids: Optional[List[Any]] = Field(default=None, description="List of frame indices for TRAKE sequence submission (alias)")
    mode: Optional[str] = Field(default="search", description="Submission mode: 'search' | 'vqa' | 'trake'")


class DresSubmitResponse(BaseModel):
    """Response model returned by POST /api/dres/submit."""
    status: bool = Field(..., description="Whether submission succeeded or was accepted by DRES")
    description: Optional[str] = Field(default=None, description="Status/message description from DRES or backend")
    evaluation_id: Optional[str] = Field(default=None, description="Target evaluation ID submitted to")
    video_id: Optional[str] = Field(default=None, description="Submitted video ID")
    timestamp: Optional[int] = Field(default=None, description="Submitted timestamp in ms")
    raw_response: Optional[Any] = Field(default=None, description="Raw response payload returned by DRES")


class DresSelectRunRequest(BaseModel):
    """
    Request body for POST /api/dres/select-run.
    Provide run_id (preferred) or run_name. At least one must be non-empty.
    """
    run_id: Optional[str] = Field(default=None, description="Exact DRES evaluation / run UUID to select")
    run_name: Optional[str] = Field(default=None, description="Exact DRES run name to select")


class DresSelectRunResponse(BaseModel):
    """Response from POST /api/dres/select-run."""
    success: bool = Field(..., description="True if the run was resolved and selected successfully")
    evaluation_id: Optional[str] = Field(default=None, description="Resolved evaluation ID now active")
    evaluation_name: Optional[str] = Field(default=None, description="Resolved evaluation name now active")
    selection_mode: Optional[str] = Field(default=None, description="'manual_run_id' | 'manual_run_name'")
    message: Optional[str] = Field(default=None, description="Human-readable result or error detail")


class DresStatusResponse(BaseModel):
    """
    Response model for GET /api/dres/status according to AIC guideline:
    {
        "connected": true,
        "evaluation": "...",
        "evaluation_id": "...",
        "run_name": "...",
        "run_id": "...",
        "selection_mode": "run_id | run_name | auto",
        "session_id": "...",
        "message": "..."
    }
    """
    connected: bool = Field(..., description="True if DRES is reachable and session is active")
    evaluation: Optional[str] = Field(default=None, description="Active evaluation name")
    evaluation_id: Optional[str] = Field(default=None, description="Active evaluation ID")
    run_name: Optional[str] = Field(default=None, description="Configured DRES_RUN_NAME (if any)")
    run_id: Optional[str] = Field(default=None, description="Configured DRES_RUN_ID (if any)")
    selection_mode: Optional[str] = Field(default=None, description="How the evaluation was selected: 'run_id' | 'run_name' | 'auto'")
    session_id: Optional[str] = Field(default=None, description="Current cached session ID")
    message: Optional[str] = Field(default=None, description="Additional status message / notes")

