"""
Async HTTP Client for DRES (Distributed Retrieval Evaluation Server) API v2.
Uses httpx.AsyncClient for high-performance asynchronous HTTP requests.
Follows official AI Challenge guidelines.
"""

import logging
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("dres.client")


class DresClientException(Exception):
    """Base exception for DRES client errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class DresAuthError(DresClientException):
    """401 Unauthorized - Session expired or invalid credentials."""
    pass


class DresForbiddenError(DresClientException):
    """403 Forbidden - Access denied or submission rejected."""
    pass


class DresNotFoundError(DresClientException):
    """404 Not Found - Evaluation or endpoint not found."""
    pass


class DresServerError(DresClientException):
    """500 Internal Server Error from DRES."""
    pass


class DresConnectionError(DresClientException):
    """Network connection failure or unreachable DRES server."""
    pass


class DresTimeoutError(DresClientException):
    """Request timed out while contacting DRES."""
    pass


class DresClient:
    """
    Client for interacting with DRES REST API v2 according to AIC guidelines.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_client(self) -> httpx.AsyncClient:
        """Create a configured AsyncClient."""
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    def _handle_response_error(self, exc: httpx.HTTPStatusError):
        """Map HTTPStatusError to domain-specific DresClientException."""
        status = exc.response.status_code
        body = exc.response.text
        msg = f"DRES returned HTTP {status}: {body}"

        if status == 401:
            raise DresAuthError(msg, status_code=status, response_body=body) from exc
        elif status == 403:
            raise DresForbiddenError(msg, status_code=status, response_body=body) from exc
        elif status == 404:
            raise DresNotFoundError(msg, status_code=status, response_body=body) from exc
        elif status >= 500:
            raise DresServerError(msg, status_code=status, response_body=body) from exc
        else:
            raise DresClientException(msg, status_code=status, response_body=body) from exc

    async def login(self, username: str, password: str) -> str:
        """
        Authenticate with DRES.
        POST /api/v2/login
        Body: {"username": "...", "password": "..."}
        Returns: sessionId (str)
        """
        url = "/api/v2/login"
        payload = {"username": username, "password": password}

        try:
            async with self._get_client() as client:
                res = await client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()

                session_id = (
                    data.get("sessionId")
                    or data.get("session_id")
                    or data.get("token")
                    or (data.get("session") if isinstance(data.get("session"), str) else None)
                )

                if not session_id:
                    raise DresClientException(f"No sessionId found in login response: {data}")

                return session_id

        except httpx.HTTPStatusError as e:
            self._handle_response_error(e)
        except httpx.TimeoutException as e:
            raise DresTimeoutError(f"Login timed out after {self.timeout}s: {e}") from e
        except httpx.RequestError as e:
            raise DresConnectionError(f"Failed to connect to DRES at {self.base_url}: {e}") from e

    async def get_evaluations(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Fetch list of evaluations.
        GET /api/v2/client/evaluation/list?session=<sessionId>
        """
        url = "/api/v2/client/evaluation/list"
        params = {"session": session_id}

        try:
            async with self._get_client() as client:
                res = await client.get(url, params=params)
                res.raise_for_status()
                data = res.json()

                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("evaluations", data.get("items", [data]))
                return []

        except httpx.HTTPStatusError as e:
            self._handle_response_error(e)
        except httpx.TimeoutException as e:
            raise DresTimeoutError(f"Fetching evaluations timed out after {self.timeout}s: {e}") from e
        except httpx.RequestError as e:
            raise DresConnectionError(f"Failed to connect to DRES at {self.base_url}: {e}") from e

    async def get_active_evaluation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Only calls /api/v2/client/evaluation/list?session=<sessionId>.
        Returns the first evaluation whose status == 'ACTIVE'.
        Fallback to the only evaluation if only one exists.
        Does not query or depend on currentTask endpoints.
        """
        evaluations = await self.get_evaluations(session_id)
        if not evaluations:
            return None

        for ev in evaluations:
            ev_status = str(ev.get("status", "")).upper()
            ev_state = str(ev.get("state", "")).upper()
            if ev_status == "ACTIVE" or ev_state == "ACTIVE":
                return ev

        # Fallback if only one evaluation exists
        if len(evaluations) == 1:
            return evaluations[0]

        return None

    async def submit_payload(self, evaluation_id: str, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit raw answerSets payload to DRES.
        POST /api/v2/submit/{evaluationId}?session=<sessionId>
        """
        url = f"/api/v2/submit/{evaluation_id}"
        params = {"session": session_id}

        try:
            async with self._get_client() as client:
                res = await client.post(url, params=params, json=payload)
                res.raise_for_status()
                return res.json()

        except httpx.HTTPStatusError as e:
            self._handle_response_error(e)
        except httpx.TimeoutException as e:
            raise DresTimeoutError(f"Submission timed out after {self.timeout}s: {e}") from e
        except httpx.RequestError as e:
            raise DresConnectionError(f"Failed to submit to DRES at {self.base_url}: {e}") from e

    async def submit_kis(
        self,
        evaluation_id: str,
        session_id: str,
        video_id: str,
        start: int,
        end: int
    ) -> Dict[str, Any]:
        """
        Submit Known-Item Search (KIS) answer.
        """
        payload = {
            "answerSets": [
                {
                    "answers": [
                        {
                            "mediaItemName": video_id,
                            "start": int(start),
                            "end": int(end)
                        }
                    ]
                }
            ]
        }
        return await self.submit_payload(evaluation_id, session_id, payload)

    async def submit_qa(
        self,
        evaluation_id: str,
        session_id: str,
        answer_text: str,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit Question Answering (QA) answer.
        """
        ans_item: Dict[str, Any] = {"text": answer_text}
        if video_id:
            ans_item["mediaItemName"] = video_id

        payload = {
            "answerSets": [
                {
                    "answers": [ans_item]
                }
            ]
        }
        return await self.submit_payload(evaluation_id, session_id, payload)

    async def submit_trake(
        self,
        evaluation_id: str,
        session_id: str,
        video_id: str,
        start: int,
        end: int,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit Targeted Retrieval and Knowledge Extraction (TraKE) answer.
        """
        ans_item: Dict[str, Any] = {
            "mediaItemName": video_id,
            "start": int(start),
            "end": int(end)
        }
        if text:
            ans_item["text"] = text

        payload = {
            "answerSets": [
                {
                    "answers": [ans_item]
                }
            ]
        }
        return await self.submit_payload(evaluation_id, session_id, payload)
