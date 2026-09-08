"""
Service layer for DRES integration.
Manages session caching, automatic re-authentication, active evaluation discovery,
and payload conversion for KIS submissions.
Follows official AI Challenge guidelines.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from dres.client import (
    DresAuthError,
    DresClient,
    DresClientException,
    DresConnectionError,
    DresForbiddenError,
    DresNotFoundError,
    DresServerError,
    DresTimeoutError,
)
from dres.config import DresSettings, get_dres_settings
from dres.schemas import DresLoginResponse, DresStatusResponse, DresSubmitResponse

logger = logging.getLogger("dres.service")


def build_vqa_payload(answer: str, video_id: str, timestamp_ms: int) -> Dict[str, Any]:
    """
    Build VQA submission payload matching official BTC format:
    "QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>"
    """
    clean_answer = str(answer).strip()
    clean_vid = str(video_id).replace(".mp4", "").strip()
    clean_ts = int(round(float(timestamp_ms))) if timestamp_ms is not None else 0
    formatted_text = f"QA-{clean_answer}-{clean_vid}-{clean_ts}"
    return {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": formatted_text
                    }
                ]
            }
        ]
    }


def build_trake_payload(video_id: str, frame_ids: List[Any]) -> Dict[str, Any]:
    """
    Build TRAKE sequence submission payload matching official BTC format:
    "TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,..."
    """
    clean_vid = str(video_id).replace(".mp4", "").strip()
    clean_frames = []
    for f in frame_ids:
        if f is None or f == "":
            continue
        if isinstance(f, dict):
            val = f.get("frame_idx") if f.get("frame_idx") is not None else f.get("frameId")
        else:
            val = f
        try:
            clean_frames.append(str(int(val)))
        except (ValueError, TypeError):
            if str(val).strip():
                clean_frames.append(str(val).strip())
    frames_str = ",".join(clean_frames)
    formatted_text = f"TR-{clean_vid}-{frames_str}"
    return {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": formatted_text
                    }
                ]
            }
        ]
    }


class DresService:
    """
    Stateful service managing DRES session tokens, evaluation discovery,
    automatic re-login, and submission dispatching.
    """

    def __init__(self, settings: Optional[DresSettings] = None):
        self.settings = settings or get_dres_settings()
        self.client = DresClient(base_url=self.settings.base_url, timeout=self.settings.timeout)
        self._session_id: Optional[str] = None
        self._cached_evaluation_id: Optional[str] = None
        self._cached_evaluation_name: Optional[str] = None
        self._selection_mode: Optional[str] = None  # 'manual_run_id' | 'manual_run_name' | 'run_id' | 'run_name' | 'auto'
        # Runtime manual override (via POST /api/dres/select-run). Takes top priority.
        self._manual_run_id: Optional[str] = None
        self._manual_run_name: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def is_configured(self) -> bool:
        """Check if username and password are provided."""
        return bool(self.settings.username and self.settings.password)

    async def get_session(self, force_refresh: bool = False) -> str:
        """
        Get a valid session ID. If none is cached or force_refresh is requested,
        authenticates with DRES and updates the cached session.
        """
        async with self._lock:
            if not self._session_id or force_refresh:
                if not self.is_configured:
                    raise DresAuthError(
                        "DRES credentials not configured. Please set DRES_USERNAME and DRES_PASSWORD in .env."
                    )
                try:
                    logger.info(f"Authenticating with DRES at {self.settings.base_url} as user '{self.settings.username}'...")
                    session_id = await self.client.login(
                        username=self.settings.username,
                        password=self.settings.password,
                    )
                    self._session_id = session_id
                    self._cached_evaluation_id = None
                    self._cached_evaluation_name = None
                    logger.info(f"Login success: sessionId='{self._session_id}'")
                except Exception as exc:
                    logger.error(f"Login failed: {exc}")
                    raise
            return self._session_id

    async def login(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> DresLoginResponse:
        """
        Force a fresh login to DRES.
        Optionally accepts override username/password.
        """
        user = username or self.settings.username
        pwd = password or self.settings.password

        if not user or not pwd:
            logger.error("Login failed: Username or password missing.")
            return DresLoginResponse(
                sessionId=None,
                status=False,
                message="Username or password missing.",
            )

        async with self._lock:
            try:
                session_id = await self.client.login(username=user, password=pwd)
                self._session_id = session_id
                self._cached_evaluation_id = None
                self._cached_evaluation_name = None
                logger.info(f"Login success: user='{user}', sessionId='{session_id}'")
                return DresLoginResponse(
                    sessionId=session_id,
                    status=True,
                    message="Login successful.",
                )
            except Exception as exc:
                logger.error(f"Login failed for user='{user}': {exc}")
                return DresLoginResponse(
                    sessionId=None,
                    status=False,
                    message=f"Login failed: {exc}",
                )

    def _extract_eval_fields(self, ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Extract (eval_id, eval_name) from a raw evaluation dict."""
        eval_id = (
            ev.get("id")
            or ev.get("evaluationId")
            or ev.get("competitionId")
            or ev.get("runId")
        )
        eval_name = ev.get("name") or ev.get("title") or str(eval_id)
        return (str(eval_id) if eval_id is not None else None,
                str(eval_name) if eval_name is not None else None)

    async def _resolve_evaluation_by_config(self, session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve the target evaluation following priority:
          0. Manual override (_manual_run_id / _manual_run_name) set via select_run()
          1. DRES_RUN_ID env var  → exact match on runId / id / evaluationId
          2. DRES_RUN_NAME env var → exact match on name; prefer RUNNING, error if ambiguous
          3. auto                 → first ACTIVE evaluation (backward-compat)

        Caches result in self._cached_evaluation_id / _name / _selection_mode.
        Raises ValueError with a clear message when configuration is invalid.
        """
        evaluations: List[Dict[str, Any]] = await self.client.get_evaluations(session_id)

        # ── Priority 0: runtime manual override (select_run API) ─────────────
        cfg_run_id = self._manual_run_id or self.settings.run_id
        cfg_run_name = self._manual_run_name if not cfg_run_id else None
        if not cfg_run_id:
            cfg_run_name = self._manual_run_name or self.settings.run_name

        # Determine selection_mode prefix for logging
        is_manual_id = bool(self._manual_run_id and cfg_run_id == self._manual_run_id)
        is_manual_name = bool(self._manual_run_name and not self._manual_run_id)

        # ── Priority 1: explicit Run ID ──────────────────────────────────────
        if cfg_run_id:
            for ev in evaluations:
                candidate_ids = [
                    str(ev.get("id", "")),
                    str(ev.get("evaluationId", "")),
                    str(ev.get("competitionId", "")),
                    str(ev.get("runId", "")),
                ]
                if cfg_run_id in candidate_ids:
                    eval_id, eval_name = self._extract_eval_fields(ev)
                    self._cached_evaluation_id = eval_id
                    self._cached_evaluation_name = eval_name
                    mode_label = "manual_run_id" if is_manual_id else "run_id"
                    self._selection_mode = mode_label
                    logger.info(
                        f"[DRES] Selection mode : {mode_label}\n"
                        f"       Target         : {cfg_run_id}\n"
                        f"       Resolved name  : {eval_name}\n"
                        f"       Status         : {ev.get('status', 'UNKNOWN')}"
                    )
                    return eval_id, eval_name
            # Run ID provided but not found — hard error
            available = [str(ev.get('id') or ev.get('evaluationId') or '') for ev in evaluations]
            source = "manual override" if is_manual_id else "DRES_RUN_ID env"
            raise ValueError(
                f"[DRES] Run ID '{cfg_run_id}' ({source}) not found in available evaluations: {available}. "
                "Check the ID or use /api/dres/select-run with a valid ID."
            )

        # ── Priority 2: explicit Run Name ────────────────────────────────────
        if cfg_run_name:
            matches = [ev for ev in evaluations if ev.get("name") == cfg_run_name]
            if not matches:
                available_names = [ev.get("name", "") for ev in evaluations]
                source = "manual override" if is_manual_name else "DRES_RUN_NAME env"
                raise ValueError(
                    f"[DRES] Run name '{cfg_run_name}' ({source}) not found in available evaluations: {available_names}. "
                    "Check the name or use /api/dres/select-run with a valid name."
                )
            if len(matches) == 1:
                ev = matches[0]
            else:
                # Multiple matches — prefer RUNNING/ACTIVE
                running = [
                    ev for ev in matches
                    if str(ev.get("status", "")).upper() in ("RUNNING", "ACTIVE")
                ]
                if len(running) == 1:
                    ev = running[0]
                else:
                    ids = [str(e.get('id') or e.get('evaluationId') or '') for e in matches]
                    raise ValueError(
                        f"[DRES] Run name '{cfg_run_name}' matches {len(matches)} evaluations "
                        f"and none is uniquely RUNNING. Please use run_id instead. Candidates: {ids}."
                    )
            eval_id, eval_name = self._extract_eval_fields(ev)
            self._cached_evaluation_id = eval_id
            self._cached_evaluation_name = eval_name
            mode_label = "manual_run_name" if is_manual_name else "run_name"
            self._selection_mode = mode_label
            logger.info(
                f"[DRES] Selection mode : {mode_label}\n"
                f"       Target         : {cfg_run_name}\n"
                f"       Resolved ID    : {eval_id}\n"
                f"       Status         : {ev.get('status', 'UNKNOWN')}"
            )
            return eval_id, eval_name

        # ── Priority 3: auto (first ACTIVE / only evaluation) ────────────────
        eval_info = await self.client.get_active_evaluation(session_id)
        if not eval_info:
            logger.warning("[DRES] Selection mode : auto — No active evaluation found.")
            self._cached_evaluation_id = None
            self._cached_evaluation_name = None
            self._selection_mode = "auto"
            return None, None

        eval_id, eval_name = self._extract_eval_fields(eval_info)
        self._cached_evaluation_id = eval_id
        self._cached_evaluation_name = eval_name
        self._selection_mode = "auto"
        logger.info(
            f"[DRES] Selection mode : auto\n"
            f"       Resolved first active evaluation\n"
            f"       ID   : {eval_id}\n"
            f"       Name : {eval_name}"
        )
        return eval_id, eval_name

    # Keep old name as alias for callers inside submit_kis / submit_vqa
    async def _resolve_active_evaluation(self, session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Alias for backward compatibility — delegates to _resolve_evaluation_by_config."""
        return await self._resolve_evaluation_by_config(session_id)

    async def select_run(
        self,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> "DresSelectRunResponse":  # noqa: F821 (forward ref resolved at runtime)
        """
        Dynamically override the active DRES evaluation without restarting the server.
        Called by POST /api/dres/select-run.
        Priority: run_id > run_name.
        The selection persists for all subsequent operations until changed.
        """
        from dres.schemas import DresSelectRunResponse

        run_id = (run_id or "").strip() or None
        run_name = (run_name or "").strip() or None

        if not run_id and not run_name:
            return DresSelectRunResponse(
                success=False,
                message="At least one of run_id or run_name must be provided.",
            )

        # Store manual override; clear cache so next submission re-resolves.
        self._manual_run_id = run_id
        self._manual_run_name = run_name if not run_id else None
        self._cached_evaluation_id = None
        self._cached_evaluation_name = None

        try:
            session_id = await self.get_session()
            eval_id, eval_name = await self._resolve_evaluation_by_config(session_id)
        except ValueError as exc:
            # Resolution failed — roll back manual override so state stays consistent
            self._manual_run_id = None
            self._manual_run_name = None
            logger.error(f"select_run failed: {exc}")
            return DresSelectRunResponse(
                success=False,
                message=str(exc),
            )
        except Exception as exc:
            self._manual_run_id = None
            self._manual_run_name = None
            logger.error(f"select_run unexpected error: {exc}")
            return DresSelectRunResponse(
                success=False,
                message=f"Unexpected error: {exc}",
            )

        if not eval_id:
            self._manual_run_id = None
            self._manual_run_name = None
            return DresSelectRunResponse(
                success=False,
                message="No matching evaluation found for the provided run_id / run_name.",
            )

        logger.info(
            f"[DRES] Manual run selected via API: id='{eval_id}', name='{eval_name}', "
            f"mode='{self._selection_mode}'"
        )
        return DresSelectRunResponse(
            success=True,
            evaluation_id=eval_id,
            evaluation_name=eval_name,
            selection_mode=self._selection_mode,
            message=f"Active evaluation switched to '{eval_name}' (id={eval_id}).",
        )

    async def get_status(self) -> DresStatusResponse:
        """
        Return the current DRES status (connected, active evaluation, evaluation_id, session_id).
        Also returns run_name, run_id, and selection_mode for diagnostics.
        Automatically retries login if session expired.
        """
        for attempt in range(self.settings.retry_count + 1):
            try:
                session_id = await self.get_session(force_refresh=(attempt > 0))
                eval_id, eval_name = await self._resolve_evaluation_by_config(session_id)

                if not eval_id:
                    return DresStatusResponse(
                        connected=True,
                        evaluation=None,
                        evaluation_id=None,
                        run_name=self.settings.run_name,
                        run_id=self.settings.run_id,
                        selection_mode=self._selection_mode,
                        session_id=session_id,
                        message="Connected to DRES, but no active evaluation found.",
                    )

                return DresStatusResponse(
                    connected=True,
                    evaluation=eval_name,
                    evaluation_id=eval_id,
                    run_name=self.settings.run_name,
                    run_id=self.settings.run_id,
                    selection_mode=self._selection_mode,
                    session_id=session_id,
                    message="Active evaluation found.",
                )

            except DresAuthError as e:
                logger.warning(f"Session expired or unauthorized during status check (attempt {attempt + 1}): {e}")
                if attempt == self.settings.retry_count:
                    return DresStatusResponse(
                        connected=False,
                        run_name=self.settings.run_name,
                        run_id=self.settings.run_id,
                        message=f"DRES authentication failed: {e}",
                    )
            except (DresConnectionError, DresTimeoutError) as e:
                logger.error(f"DRES connection error: {e}")
                return DresStatusResponse(
                    connected=False,
                    run_name=self.settings.run_name,
                    run_id=self.settings.run_id,
                    message=f"Cannot reach DRES at {self.settings.base_url}: {e}",
                )
            except ValueError as e:
                # Configuration validation error (wrong run_id / run_name)
                logger.error(str(e))
                return DresStatusResponse(
                    connected=True,
                    run_name=self.settings.run_name,
                    run_id=self.settings.run_id,
                    message=str(e),
                )
            except Exception as e:
                logger.error(f"Unexpected error checking DRES status: {e}")
                return DresStatusResponse(
                    connected=False,
                    run_name=self.settings.run_name,
                    run_id=self.settings.run_id,
                    message=f"Error checking status: {e}",
                )

        return DresStatusResponse(
            connected=False,
            run_name=self.settings.run_name,
            run_id=self.settings.run_id,
            message="Failed to connect to DRES after retries.",
        )

    async def submit_kis(
        self,
        video_id: str,
        timestamp: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        text: Optional[str] = None
    ) -> DresSubmitResponse:
        """
        Submit a KIS item to DRES.
        Auto-converts video_id and timestamp (ms) into DRES payload.
        Handles 401 session expiration by re-logging in and retrying once.
        """
        s_ms = int(start if start is not None else timestamp)
        e_ms = int(end if end is not None else timestamp)

        for attempt in range(self.settings.retry_count + 1):
            try:
                session_id = await self.get_session(force_refresh=(attempt > 0))

                # Ensure evaluation ID is available
                eval_id = self._cached_evaluation_id
                if not eval_id:
                    eval_id, _ = await self._resolve_active_evaluation(session_id)
                    if not eval_id:
                        err_msg = "No active evaluation found on DRES to submit answer to."
                        logger.error(f"Submission failed: {err_msg}")
                        return DresSubmitResponse(
                            status=False,
                            description=err_msg,
                            video_id=video_id,
                            timestamp=timestamp,
                        )

                logger.info(
                    f"Submitting KIS to DRES eval='{eval_id}': "
                    f"mediaItemName='{video_id}', start={s_ms}, end={e_ms}"
                )

                if text:
                    res = await self.client.submit_trake(
                        evaluation_id=eval_id,
                        session_id=session_id,
                        video_id=video_id,
                        start=s_ms,
                        end=e_ms,
                        text=text,
                    )
                else:
                    res = await self.client.submit_kis(
                        evaluation_id=eval_id,
                        session_id=session_id,
                        video_id=video_id,
                        start=s_ms,
                        end=e_ms,
                    )

                logger.info(
                    f"Submission success: eval='{eval_id}', "
                    f"video_id='{video_id}', timestamp={timestamp}, result={res}"
                )

                description = (
                    res.get("description")
                    or res.get("status")
                    or res.get("message")
                    or "Submission accepted by DRES."
                )

                raw_status = res.get("status")
                if isinstance(raw_status, bool):
                    is_ok = raw_status
                else:
                    status_val = str(raw_status if raw_status is not None else "true").upper()
                    is_ok = status_val not in ("WRONG", "ERROR", "FAILED", "FALSE")

                description = res.get("submission") or res.get("description") or res.get("message") or (
                    "Submission accepted by DRES." if is_ok else "Submission rejected by DRES."
                )

                return DresSubmitResponse(
                    status=is_ok,
                    description=str(description),
                    evaluation_id=str(eval_id),
                    video_id=video_id,
                    timestamp=timestamp,
                    raw_response=res,
                )

            except DresAuthError as exc:
                logger.warning(
                    f"Session expired during submission (attempt {attempt + 1}/{self.settings.retry_count + 1}): {exc}"
                )
                if attempt == self.settings.retry_count:
                    logger.error(f"Submission failed due to authentication: {exc}")
                    return DresSubmitResponse(
                        status=False,
                        description=f"DRES auth error: {exc}",
                        evaluation_id=self._cached_evaluation_id,
                        video_id=video_id,
                        timestamp=timestamp,
                    )
            except (DresForbiddenError, DresNotFoundError, DresServerError, DresClientException) as exc:
                logger.error(f"Submission failed with API error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"DRES API error: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                    timestamp=timestamp,
                )
            except (DresConnectionError, DresTimeoutError) as exc:
                logger.error(f"Submission failed with network error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"Network error contacting DRES: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                    timestamp=timestamp,
                )
            except Exception as exc:
                logger.error(f"Submission failed with unexpected error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"Unexpected error: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                    timestamp=timestamp,
                )

        return DresSubmitResponse(
            status=False,
            description="Submission failed after retry attempts.",
            evaluation_id=self._cached_evaluation_id,
            video_id=video_id,
            timestamp=timestamp,
        )

    async def submit_vqa(
        self,
        answer: str,
        video_id: str,
        timestamp_ms: int,
    ) -> DresSubmitResponse:
        """
        Submit a VQA answer to DRES using standard BTC format QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>.
        Auto-refreshes session and retries if session expired.
        """
        payload = build_vqa_payload(answer=answer, video_id=video_id, timestamp_ms=timestamp_ms)

        for attempt in range(self.settings.retry_count + 1):
            try:
                session_id = await self.get_session(force_refresh=(attempt > 0))

                eval_id = self._cached_evaluation_id
                if not eval_id:
                    eval_id, _ = await self._resolve_active_evaluation(session_id)
                    if not eval_id:
                        err_msg = "No active evaluation found on DRES to submit answer to."
                        logger.error(f"VQA submission failed: {err_msg}")
                        return DresSubmitResponse(
                            status=False,
                            description=err_msg,
                            video_id=video_id,
                            timestamp=timestamp_ms,
                        )

                logger.info(
                    f"Submitting VQA to DRES eval='{eval_id}': "
                    f"answer='{answer}', video_id='{video_id}', timestamp={timestamp_ms}"
                )

                res = await self.client.submit_payload(
                    evaluation_id=eval_id,
                    session_id=session_id,
                    payload=payload,
                )

                logger.info(
                    f"VQA submission response: eval='{eval_id}', "
                    f"video_id='{video_id}', timestamp={timestamp_ms}, result={res}"
                )

                raw_status = res.get("status")
                if isinstance(raw_status, bool):
                    is_ok = raw_status
                else:
                    status_val = str(raw_status if raw_status is not None else "true").upper()
                    is_ok = status_val not in ("WRONG", "ERROR", "FAILED", "FALSE")

                description = res.get("submission") or res.get("description") or res.get("message") or (
                    "Submission accepted by DRES." if is_ok else "Submission rejected by DRES."
                )

                return DresSubmitResponse(
                    status=is_ok,
                    description=str(description),
                    evaluation_id=str(eval_id),
                    video_id=video_id,
                    timestamp=timestamp_ms,
                    raw_response=res,
                )

            except DresAuthError as exc:
                logger.warning(
                    f"Session expired during VQA submission (attempt {attempt + 1}/{self.settings.retry_count + 1}): {exc}"
                )
                if attempt == self.settings.retry_count:
                    logger.error(f"VQA submission failed due to authentication: {exc}")
                    return DresSubmitResponse(
                        status=False,
                        description=f"DRES auth error: {exc}",
                        evaluation_id=self._cached_evaluation_id,
                        video_id=video_id,
                        timestamp=timestamp_ms,
                    )
            except (DresForbiddenError, DresNotFoundError, DresServerError, DresClientException) as exc:
                logger.error(f"VQA submission failed with API error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"DRES API error: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                    timestamp=timestamp_ms,
                )
            except (DresConnectionError, DresTimeoutError) as exc:
                logger.error(f"VQA submission failed with network error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"Network error contacting DRES: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                    timestamp=timestamp_ms,
                )
            except Exception as exc:
                logger.error(f"VQA submission failed with unexpected error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"Unexpected error: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                    timestamp=timestamp_ms,
                )

        return DresSubmitResponse(
            status=False,
            description="Submission failed after retry attempts.",
            evaluation_id=self._cached_evaluation_id,
            video_id=video_id,
            timestamp=timestamp_ms,
        )

    async def submit_trake(
        self,
        video_id: str,
        frame_ids: List[Any],
    ) -> DresSubmitResponse:
        """
        Submit a TRAKE sequence to DRES using standard BTC format:
        TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...
        Auto-refreshes session and retries if session expired.
        """
        payload = build_trake_payload(video_id=video_id, frame_ids=frame_ids)
        formatted_text = payload["answerSets"][0]["answers"][0]["text"]

        for attempt in range(self.settings.retry_count + 1):
            try:
                session_id = await self.get_session(force_refresh=(attempt > 0))

                eval_id = self._cached_evaluation_id
                if not eval_id:
                    eval_id, _ = await self._resolve_active_evaluation(session_id)
                    if not eval_id:
                        err_msg = "No active evaluation found on DRES to submit answer to."
                        logger.error(f"TRAKE submission failed: {err_msg}")
                        return DresSubmitResponse(
                            status=False,
                            description=err_msg,
                            video_id=video_id,
                        )

                logger.info(
                    f"Submitting TRAKE to DRES eval='{eval_id}': "
                    f"formatted_text='{formatted_text}'"
                )

                res = await self.client.submit_payload(
                    evaluation_id=eval_id,
                    session_id=session_id,
                    payload=payload,
                )

                logger.info(
                    f"TRAKE submission response: eval='{eval_id}', "
                    f"text='{formatted_text}', result={res}"
                )

                raw_status = res.get("status")
                if isinstance(raw_status, bool):
                    is_ok = raw_status
                else:
                    status_val = str(raw_status if raw_status is not None else "true").upper()
                    is_ok = status_val not in ("WRONG", "ERROR", "FAILED", "FALSE")

                description = res.get("submission") or res.get("description") or res.get("message") or (
                    "Submission accepted by DRES." if is_ok else "Submission rejected by DRES."
                )

                return DresSubmitResponse(
                    status=is_ok,
                    description=str(description),
                    evaluation_id=str(eval_id),
                    video_id=video_id,
                    raw_response=res,
                )

            except DresAuthError as exc:
                logger.warning(
                    f"Session expired during TRAKE submission (attempt {attempt + 1}/{self.settings.retry_count + 1}): {exc}"
                )
                if attempt == self.settings.retry_count:
                    logger.error(f"TRAKE submission failed due to authentication: {exc}")
                    return DresSubmitResponse(
                        status=False,
                        description=f"DRES auth error: {exc}",
                        evaluation_id=self._cached_evaluation_id,
                        video_id=video_id,
                    )
            except (DresForbiddenError, DresNotFoundError, DresServerError, DresClientException) as exc:
                logger.error(f"TRAKE submission failed with API error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"DRES API error: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                )
            except (DresConnectionError, DresTimeoutError) as exc:
                logger.error(f"TRAKE submission failed with network error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"Network error contacting DRES: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                )
            except Exception as exc:
                logger.error(f"TRAKE submission failed with unexpected error: {exc}")
                return DresSubmitResponse(
                    status=False,
                    description=f"Unexpected error: {exc}",
                    evaluation_id=self._cached_evaluation_id,
                    video_id=video_id,
                )

        return DresSubmitResponse(
            status=False,
            description="Submission failed after retry attempts.",
            evaluation_id=self._cached_evaluation_id,
            video_id=video_id,
        )


# Singleton instance provider
_dres_service_instance: Optional[DresService] = None


def get_dres_service() -> DresService:
    """Provide singleton instance of DresService."""
    global _dres_service_instance
    if _dres_service_instance is None:
        _dres_service_instance = DresService()
    return _dres_service_instance
