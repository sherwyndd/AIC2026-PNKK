"""
DRES (Distributed Retrieval Evaluation Server) Package for AIC 2026.
"""

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
from dres.router import router as dres_router
from dres.schemas import (
    DresAnswerItem,
    DresAnswerSet,
    DresLoginRequest,
    DresLoginResponse,
    DresStatusResponse,
    DresSubmissionPayload,
    DresSubmitRequest,
    DresSubmitResponse,
)
from dres.service import DresService, build_vqa_payload, get_dres_service

__all__ = [
    "DresSettings",
    "get_dres_settings",
    "DresClient",
    "DresService",
    "get_dres_service",
    "build_vqa_payload",
    "dres_router",
    "DresLoginRequest",
    "DresLoginResponse",
    "DresAnswerItem",
    "DresAnswerSet",
    "DresSubmissionPayload",
    "DresSubmitRequest",
    "DresSubmitResponse",
    "DresStatusResponse",
    "DresClientException",
    "DresAuthError",
    "DresForbiddenError",
    "DresNotFoundError",
    "DresServerError",
    "DresConnectionError",
    "DresTimeoutError",
]
