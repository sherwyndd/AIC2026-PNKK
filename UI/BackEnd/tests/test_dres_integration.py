"""
Unit tests for DRES integration module.
Tests schemas, service state caching, retry mechanisms, and API endpoints.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dres.client import DresAuthError, DresClient
from dres.config import DresSettings
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
from dres.service import DresService, build_vqa_payload


async def test_schemas():
    """Test schema parsing and serialization."""
    # Test Submit Request with standard search & VQA fields
    req = DresSubmitRequest(video_id="L26_V320", timestamp=145000)
    assert req.video_id == "L26_V320"
    assert req.timestamp == 145000

    # Test Payload building
    item = DresAnswerItem(mediaItemName=req.video_id, start=req.timestamp, end=req.timestamp)
    answer_set = DresAnswerSet(answers=[item])
    payload = DresSubmissionPayload(answerSets=[answer_set])
    payload_dict = payload.model_dump()
    assert payload_dict["answerSets"][0]["answers"][0]["mediaItemName"] == "L26_V320"
    assert payload_dict["answerSets"][0]["answers"][0]["start"] == 145000
    assert payload_dict["answerSets"][0]["answers"][0]["end"] == 145000

    # Test VQA payload helper
    vqa_payload = build_vqa_payload("TIKTOK", "L21_V006", 866933)
    assert vqa_payload == {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": "QA-TIKTOK-L21_V006-866933"
                    }
                ]
            }
        ]
    }
    print("✓ Schema & VQA payload test passed")


async def test_service_login_and_cache():
    """Test login caching and retry logic in DresService."""
    settings = DresSettings()
    settings.base_url = "http://192.168.28.151:5000"
    settings.username = "test_user"
    settings.password = "test_pass"
    settings.retry_count = 1

    service = DresService(settings=settings)

    # Mock client methods
    service.client.login = AsyncMock(return_value="mock_session_123")
    service.client.get_active_evaluation = AsyncMock(return_value={
        "id": "eval_001",
        "name": "AIC 2026 Evaluation",
        "status": "ACTIVE"
    })
    service.client.submit_kis = AsyncMock(return_value={
        "status": "CORRECT",
        "description": "Correct keyframe submitted"
    })
    service.client.submit_payload = AsyncMock(return_value={
        "status": "CORRECT",
        "description": "Correct VQA submitted"
    })

    # 1. Test get_status
    status = await service.get_status()
    assert status.connected is True
    assert status.evaluation == "AIC 2026 Evaluation"
    assert status.evaluation_id == "eval_001"
    assert status.session_id == "mock_session_123"
    print("✓ get_status test passed")

    # 2. Test submit_kis
    submit_res = await service.submit_kis(video_id="L26_V320", timestamp=145000)
    assert submit_res.status is True
    assert submit_res.video_id == "L26_V320"
    assert submit_res.evaluation_id == "eval_001"
    service.client.submit_kis.assert_called_once_with(
        evaluation_id="eval_001",
        session_id="mock_session_123",
        video_id="L26_V320",
        start=145000,
        end=145000
    )
    print("✓ submit_kis test passed")

    # 3. Test submit_vqa
    vqa_res = await service.submit_vqa(answer="TIKTOK", video_id="L21_V006", timestamp_ms=866933)
    assert vqa_res.status is True
    assert vqa_res.video_id == "L21_V006"
    assert vqa_res.timestamp == 866933
    service.client.submit_payload.assert_called_once_with(
        evaluation_id="eval_001",
        session_id="mock_session_123",
        payload={
            "answerSets": [
                {
                    "answers": [
                        {
                            "text": "QA-TIKTOK-L21_V006-866933"
                        }
                    ]
                }
            ]
        }
    )
    print("✓ submit_vqa test passed")

    # 4. Test automatic retry on 401 session expiration
    service.client.submit_kis = AsyncMock(side_effect=[
        DresAuthError("401 Session expired"),
        {"status": "CORRECT", "description": "Success on retry"}
    ])
    service.client.login = AsyncMock(return_value="mock_new_session_456")

    retry_res = await service.submit_kis(video_id="L26_V320", timestamp=145000)
    assert retry_res.status is True
    assert service.client.login.call_count >= 1
    print("✓ auto-relogin retry test passed")


async def main():
    await test_schemas()
    await test_service_login_and_cache()
    print("\nAll DRES integration tests PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
