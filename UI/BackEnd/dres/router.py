"""
FastAPI Router for DRES Integration.
Provides endpoints:
- GET  /api/dres/status
- POST /api/dres/login
- POST /api/dres/submit
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from dres.schemas import (
    DresLoginRequest,
    DresLoginResponse,
    DresSelectRunRequest,
    DresSelectRunResponse,
    DresStatusResponse,
    DresSubmitRequest,
    DresSubmitResponse,
)
from dres.service import DresService, get_dres_service

logger = logging.getLogger("dres.router")

router = APIRouter(prefix="/api/dres", tags=["DRES"])


@router.get("/status", response_model=DresStatusResponse)
async def get_dres_status(service: DresService = Depends(get_dres_service)):
    """
    Get current DRES connection status, active evaluation, running task, and remaining time.
    Response format:
    {
        "connected": true,
        "evaluation": "...",
        "task": "...",
        "time_left": 120.0
    }
    """
    return await service.get_status()


@router.post("/login", response_model=DresLoginResponse)
async def dres_login(
    payload: Optional[DresLoginRequest] = None,
    service: DresService = Depends(get_dres_service)
):
    """
    Force a fresh login to DRES.
    If username/password are omitted in body, uses environment credentials (DRES_USERNAME, DRES_PASSWORD).
    """
    username = payload.username if payload else None
    password = payload.password if payload else None
    
    result = await service.login(username=username, password=password)
    if not result.status:
        # Return 401 or 500 status if login completely failed
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.message or "DRES Login failed",
        )
    return result


@router.post("/submit", response_model=DresSubmitResponse)
async def dres_submit(
    req: DresSubmitRequest,
    service: DresService = Depends(get_dres_service)
):
    """
    Submit a Search/KIS, VQA, or TRAKE answer to DRES.
    - If mode == "trake" or frames/frame_ids is provided:
        TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...
    - If mode == "vqa" or answer is provided:
        QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>
    - Otherwise submits standard KIS answer.
    """
    if req.mode == "trake" or (req.frames is not None) or (req.frame_ids is not None):
        raw_frames = req.frames if req.frames is not None else (req.frame_ids if req.frame_ids is not None else [])
        if isinstance(raw_frames, str):
            raw_frames = [f.strip() for f in raw_frames.split(",") if f.strip()]
        return await service.submit_trake(
            video_id=req.video_id,
            frame_ids=raw_frames,
        )

    ts_ms = req.timestamp_ms if req.timestamp_ms is not None else (req.timestamp if req.timestamp is not None else 0)
    
    if req.mode == "vqa" or (req.answer and req.answer.strip()):
        answer_text = (req.answer or req.text or "").strip()
        return await service.submit_vqa(
            answer=answer_text,
            video_id=req.video_id,
            timestamp_ms=ts_ms,
        )

    return await service.submit_kis(
        video_id=req.video_id,
        timestamp=ts_ms,
        start=req.start,
        end=req.end,
        text=req.text,
    )


@router.post("/vqa/submit", response_model=DresSubmitResponse)
@router.post("/submit/vqa", response_model=DresSubmitResponse)
async def dres_submit_vqa(
    req: DresSubmitRequest,
    service: DresService = Depends(get_dres_service)
):
    """
    Direct endpoint to submit a VQA answer to DRES using standard BTC format:
    QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>
    """
    ts_ms = req.timestamp_ms if req.timestamp_ms is not None else (req.timestamp if req.timestamp is not None else 0)
    answer_text = (req.answer or req.text or "").strip()
    return await service.submit_vqa(
        answer=answer_text,
        video_id=req.video_id,
        timestamp_ms=ts_ms,
    )


@router.post("/trake/submit", response_model=DresSubmitResponse)
@router.post("/submit/trake", response_model=DresSubmitResponse)
async def dres_submit_trake(
    req: DresSubmitRequest,
    service: DresService = Depends(get_dres_service)
):
    """
    Direct endpoint to submit a TRAKE sequence to DRES using standard BTC format:
    TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...
    """
    raw_frames = req.frames if req.frames is not None else (req.frame_ids if req.frame_ids is not None else [])
    if isinstance(raw_frames, str):
        raw_frames = [f.strip() for f in raw_frames.split(",") if f.strip()]
    return await service.submit_trake(
        video_id=req.video_id,
        frame_ids=raw_frames,
    )


@router.post(
    "/select-run",
    response_model=DresSelectRunResponse,
    summary="Select active DRES evaluation at runtime",
    description=(
        "Dynamically switch the active DRES evaluation without restarting the server.\n\n"
        "**Priority**: `run_id` > `run_name`. At least one must be provided.\n\n"
        "The selection persists immediately for all subsequent `/submit` and `/status` calls.\n\n"
        "Intended for development / testing — no `.env` edit or restart required."
    ),
)
async def dres_select_run(
    req: DresSelectRunRequest,
    service: DresService = Depends(get_dres_service)
):
    """
    Switch the active DRES evaluation on the fly.

    - Provide **run_id** (UUID) for exact ID match.
    - Provide **run_name** for name match (prefers RUNNING if multiple share the same name).
    - At least one field must be non-empty.
    """
    result = await service.select_run(
        run_id=req.run_id,
        run_name=req.run_name,
    )
    return result
