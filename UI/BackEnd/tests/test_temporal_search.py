"""
Unit tests & Benchmark for Stage-based Temporal Search (Multi-Event Sequential KIS) matching TRAKE.

Covers:
1. Standard 2-stage sequential temporal search with matching timestamps.
2. Multi-subquery RRF fusion within stage (text + text + image).
3. Chronologically inverted candidates rejection (stage 2 timestamp < stage 1 timestamp).
4. Pruning: video missing any stage is pruned.
5. max_gap_seconds constraint verification.
6. Action-Distance NMS extracts multiple distinct sequences for the same video.
7. Global context bonus calculation.
8. FastAPI /api/v1/temporal-search/solve endpoint integration test.
"""

import time
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from fastapi.testclient import TestClient
from fastapi import FastAPI

from search.temporal_search import (
    StageQuery,
    StageSubQuery,
    TemporalSearchRequest,
    TemporalSearchResponse,
    StageEventResult,
    VideoResult,
    _search_all_stages,
    _fuse_engine_results,
    _group_and_prune_by_video,
    _beam_search_dp_alignment,
    solve_temporal_search,
    DEFAULT_RRF_K,
    DEFAULT_CONTEXT_BONUS_WEIGHT,
    temporal_router
)

# Test FastAPI App
test_app = FastAPI()
test_app.include_router(temporal_router)
client = TestClient(test_app)


class TestTemporalSearch(unittest.TestCase):

    def test_1_simple_two_stage_sequential_match(self):
        """
        Test 1: 2 stage, có 1 video match hoàn hảo theo thứ tự thời gian.
        Video 'V001' có stage 0 ở ts=10.0, stage 1 ở ts=20.0 (hợp lệ).
        Video 'V002' có stage 0 ở ts=30.0, stage 1 ở ts=15.0 (không hợp lệ).
        """
        stages = [
            StageQuery(stage_index=0, text_query="A red car arrives"),
            StageQuery(stage_index=1, text_query="A man steps out of car")
        ]

        mock_stage_engine_results = {
            0: {
                "text_0": {
                    "results": [
                        {"video_id": "V001", "frame_idx": 100, "timestamp": 10.0, "score": 0.8},
                        {"video_id": "V002", "frame_idx": 300, "timestamp": 30.0, "score": 0.9}
                    ],
                    "label": "text",
                    "weight": 1.0
                }
            },
            1: {
                "text_0": {
                    "results": [
                        {"video_id": "V001", "frame_idx": 200, "timestamp": 20.0, "score": 0.75},
                        {"video_id": "V002", "frame_idx": 150, "timestamp": 15.0, "score": 0.85}
                    ],
                    "label": "text",
                    "weight": 1.0
                }
            }
        }

        # Step 2: Fuse
        stage_candidates = _fuse_engine_results(stages, mock_stage_engine_results, top_k_per_stage=10, rrf_k=60)
        self.assertEqual(len(stage_candidates[0]), 2)
        self.assertEqual(len(stage_candidates[1]), 2)

        # Step 3: Group & Prune
        grouped = _group_and_prune_by_video(stages, stage_candidates)
        self.assertIn("V001", grouped)
        self.assertIn("V002", grouped)

        # Step 4 & 5: Beam Search DP Alignment
        # V001 should succeed
        res_v1 = _beam_search_dp_alignment("V001", stages, grouped["V001"])
        self.assertEqual(len(res_v1), 1)
        self.assertEqual(res_v1[0].video_id, "V001")
        self.assertEqual(len(res_v1[0].timeline), 2)
        self.assertEqual(res_v1[0].timeline[0].frame_idx, 100)
        self.assertEqual(res_v1[0].timeline[0].timestamp, 10.0)
        self.assertEqual(res_v1[0].timeline[1].frame_idx, 200)
        self.assertEqual(res_v1[0].timeline[1].timestamp, 20.0)
        self.assertLess(res_v1[0].timeline[0].timestamp, res_v1[0].timeline[1].timestamp)
        self.assertEqual(res_v1[0].time_span, "00:10 -> 00:20")
        self.assertEqual(res_v1[0].duration_sec, 10.0)

        # V002 should yield no valid sequence (timestamp 30.0 > 15.0)
        res_v2 = _beam_search_dp_alignment("V002", stages, grouped["V002"])
        self.assertEqual(len(res_v2), 0)

    def test_2_multi_subquery_rrf_in_stage(self):
        """
        Test 2: Stage 0 có nhiều điều kiện: RRF(text1, text2, image).
        Kiểm tra điểm RRF và matched_sources.
        """
        stage0 = StageQuery(
            stage_index=0,
            queries=[
                StageSubQuery(type="text", text="car on street", weight=1.0),
                StageSubQuery(type="text", text="traffic light", weight=0.8),
                StageSubQuery(type="image", query="L21_V001_100", weight=1.5)
            ]
        )
        stages = [stage0]

        mock_stage_engine_results = {
            0: {
                "text_0": {
                    "results": [{"video_id": "V001", "frame_idx": 100, "timestamp": 10.0, "score": 0.9}],
                    "label": "text",
                    "weight": 1.0
                },
                "text_1": {
                    "results": [{"video_id": "V001", "frame_idx": 100, "timestamp": 10.0, "score": 0.8}],
                    "label": "text",
                    "weight": 0.8
                },
                "image_2": {
                    "results": [{"video_id": "V001", "frame_idx": 100, "timestamp": 10.0, "score": 0.95}],
                    "label": "image",
                    "weight": 1.5
                }
            }
        }

        fused = _fuse_engine_results(stages, mock_stage_engine_results, top_k_per_stage=10, rrf_k=60)
        self.assertEqual(len(fused[0]), 1)
        item = fused[0][0]
        self.assertEqual(item.video_id, "V001")
        self.assertEqual(item.frame_idx, 100)
        # Expected Normalized RRF score = 1.0*(61/61) + 0.8*(61/61) + 1.5*(61/61) = 3.3
        expected_score = round(1.0 + 0.8 + 1.5, 6)
        self.assertAlmostEqual(item.score, expected_score, places=5)
        self.assertIn("text", item.matched_sources)
        self.assertIn("image", item.matched_sources)

    def test_3_chronologically_inverted_candidates_rejected(self):
        """
        Test 3: candidate có timestamp bị đảo ngược -> bị loại.
        """
        stages = [
            StageQuery(stage_index=0, text_query="Sunrise"),
            StageQuery(stage_index=1, text_query="Sunset")
        ]

        video_cands = {
            0: [StageEventResult(event_id="Stage 1", stage_index=0, video_id="V_INV", keyframe_id="k1", frame_idx=1000, timestamp=100.0, image_path="p1", score=0.015)],
            1: [StageEventResult(event_id="Stage 2", stage_index=1, video_id="V_INV", keyframe_id="k2", frame_idx=500, timestamp=50.0, image_path="p2", score=0.016)]
        }

        res = _beam_search_dp_alignment("V_INV", stages, video_cands)
        self.assertEqual(len(res), 0, "Inverted timestamp candidates must yield 0 sequences")

    def test_4_prune_missing_stages(self):
        """
        Test 4: Video thiếu bất kỳ stage nào phải bị loại ở bước Group & Prune.
        """
        stages = [
            StageQuery(stage_index=0, text_query="Event 0"),
            StageQuery(stage_index=1, text_query="Event 1"),
            StageQuery(stage_index=2, text_query="Event 2")
        ]

        stage_candidates = {
            0: [StageEventResult(event_id="Stage 1", stage_index=0, video_id="V_MISS", keyframe_id="k0", frame_idx=10, timestamp=5.0, image_path="p0", score=0.016)],
            1: [],
            2: [StageEventResult(event_id="Stage 3", stage_index=2, video_id="V_MISS", keyframe_id="k2", frame_idx=50, timestamp=25.0, image_path="p2", score=0.015)]
        }

        grouped = _group_and_prune_by_video(stages, stage_candidates)
        self.assertNotIn("V_MISS", grouped, "Video missing any stage must be pruned")

    def test_5_max_gap_seconds_constraint(self):
        """
        Test 5: Kiểm tra ràng buộc max_gap_seconds.
        Stage 0 ở t=10.0s, Stage 1 ở t=30.0s (delta=20s).
        - Nếu max_gap_seconds=15.0s -> delta=20s > 15s -> bị loại.
        - Nếu max_gap_seconds=25.0s -> delta=20s <= 25s -> hợp lệ.
        """
        stages_strict = [
            StageQuery(stage_index=0, text_query="Start", max_gap_seconds=15.0),
            StageQuery(stage_index=1, text_query="End")
        ]
        video_cands = {
            0: [StageEventResult(event_id="Stage 1", stage_index=0, video_id="V_GAP", keyframe_id="k0", frame_idx=100, timestamp=10.0, image_path="p0", score=0.02)],
            1: [StageEventResult(event_id="Stage 2", stage_index=1, video_id="V_GAP", keyframe_id="k1", frame_idx=300, timestamp=30.0, image_path="p1", score=0.02)]
        }

        res_strict = _beam_search_dp_alignment("V_GAP", stages_strict, video_cands)
        self.assertEqual(len(res_strict), 0, "Gap 20s must exceed max_gap 15s")

        stages_relaxed = [
            StageQuery(stage_index=0, text_query="Start", max_gap_seconds=25.0),
            StageQuery(stage_index=1, text_query="End")
        ]
        res_relaxed = _beam_search_dp_alignment("V_GAP", stages_relaxed, video_cands)
        self.assertEqual(len(res_relaxed), 1, "Gap 20s must pass max_gap 25s")

    def test_6_action_distance_nms_distinct_sequences(self):
        """
        Test 6: Video có các chuỗi sự kiện khác nhau ở xa nhau (>10s) -> trích xuất tối đa các sequence hợp lệ.
        """
        stages = [
            StageQuery(stage_index=0, text_query="Action 1"),
            StageQuery(stage_index=1, text_query="Action 2")
        ]

        # Chuỗi 1: t0=10.0, t1=20.0
        # Chuỗi 2: t0=60.0, t1=70.0 (cách chuỗi 1 > 10s)
        video_cands = {
            0: [
                StageEventResult(event_id="Stage 1", stage_index=0, video_id="V_MULTI", keyframe_id="k0_1", frame_idx=100, timestamp=10.0, image_path="p1", score=0.02),
                StageEventResult(event_id="Stage 1", stage_index=0, video_id="V_MULTI", keyframe_id="k0_2", frame_idx=600, timestamp=60.0, image_path="p2", score=0.019)
            ],
            1: [
                StageEventResult(event_id="Stage 2", stage_index=1, video_id="V_MULTI", keyframe_id="k1_1", frame_idx=200, timestamp=20.0, image_path="p3", score=0.02),
                StageEventResult(event_id="Stage 2", stage_index=1, video_id="V_MULTI", keyframe_id="k1_2", frame_idx=700, timestamp=70.0, image_path="p4", score=0.019)
            ]
        }

        res = _beam_search_dp_alignment("V_MULTI", stages, video_cands, max_sequences=3)
        self.assertGreaterEqual(len(res), 2, "Action-Distance NMS should return distinct sequences")
        self.assertEqual(res[0].sequence_id, 1)
        self.assertEqual(res[1].sequence_id, 2)

    def test_7_fastapi_endpoint_solve(self):
        """
        Test 7: Gọi API POST /api/v1/temporal-search/solve qua TestClient.
        """
        payload = {
            "stages": [
                {
                    "stage_index": 0,
                    "queries": [
                        {"type": "text", "text": "police car on highway", "weight": 1.0}
                    ]
                },
                {
                    "stage_index": 1,
                    "queries": [
                        {"type": "text", "text": "car chase sequence", "weight": 1.0}
                    ]
                }
            ],
            "global_context": "action movie pursuit",
            "context_bonus_weight": 0.15,
            "top_k_videos": 5
        }

        with patch("search.temporal_search._ensure_siglip_loaded"), \
             patch("search.temporal_search.search_semantic") as mock_sem:
            mock_sem.side_effect = lambda query, limit, vlist=None: [
                {"video_id": "V_API_1", "frame_idx": 100, "timestamp": 10.0, "score": 0.85, "image_path": "V_API_1/k1.jpg"},
                {"video_id": "V_API_1", "frame_idx": 200, "timestamp": 25.0, "score": 0.80, "image_path": "V_API_1/k2.jpg"}
            ]

            response = client.post("/api/v1/temporal-search/solve", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("videos", data)
            self.assertTrue(len(data["videos"]) > 0)
            v0 = data["videos"][0]
            self.assertEqual(v0["video_id"], "V_API_1")
            self.assertIn("timeline", v0)
            self.assertIn("frame_chain", v0)
            self.assertIn("time_span", v0)
            self.assertIn("duration_sec", v0)
            self.assertIn("global_score", v0)

    def test_8_fuse_text_image_formula(self):
        """
        Test 8: Kiểm tra công thức _fuse_text_image f_fusion = 0.75 * text + 0.25 * image.
        """
        from search.temporal_search import _fuse_text_image

        text_results = [
            {"video_id": "V1", "frame_idx": 100, "score": 0.80, "image_path": "V1/100.jpg"},
            {"video_id": "V1", "frame_idx": 200, "score": 0.60, "image_path": "V1/200.jpg"},
        ]
        image_results = [
            {"video_id": "V1", "frame_idx": 100, "score": 0.40, "image_path": "V1/100.jpg"},
            {"video_id": "V1", "frame_idx": 300, "score": 0.80, "image_path": "V1/300.jpg"},
        ]

        fused = _fuse_text_image(text_results, image_results)
        self.assertEqual(len(fused), 3)

        # Doc (V1, 100): 0.75 * 0.80 + 0.25 * 0.40 = 0.60 + 0.10 = 0.70
        doc100 = next(d for d in fused if d["frame_idx"] == 100)
        self.assertAlmostEqual(doc100["score"], 0.70, places=5)

        # Doc (V1, 200): 0.75 * 0.60 + 0.25 * 0.00 = 0.45
        doc200 = next(d for d in fused if d["frame_idx"] == 200)
        self.assertAlmostEqual(doc200["score"], 0.45, places=5)

        # Doc (V1, 300): 0.75 * 0.00 + 0.25 * 0.80 = 0.20
        doc300 = next(d for d in fused if d["frame_idx"] == 300)
        self.assertAlmostEqual(doc300["score"], 0.20, places=5)

        # Thứ tự ranking phải giảm dần
        self.assertEqual(fused[0]["frame_idx"], 100)
        self.assertEqual(fused[1]["frame_idx"], 200)
        self.assertEqual(fused[2]["frame_idx"], 300)

    def test_9_stage_with_text_and_image_fused_as_one_run(self):
        """
        Test 9: Khi 1 stage có cả text_query và image_query -> được fuse thành 1 run duy nhất (text_image_fusion) trước RRF.
        """
        from search.temporal_search import _fuse_engine_results

        stages = [
            StageQuery(
                stage_index=0,
                queries=[
                    StageSubQuery(type="text", text="drone show", weight=1.0),
                    StageSubQuery(type="image", query="L22_V031_100", weight=1.0),
                    StageSubQuery(type="ocr", text="VIETNAM", weight=1.0),
                ]
            )
        ]

        mock_stage_engine_results = {
            0: {
                "text_0": {
                    "results": [{"video_id": "V1", "frame_idx": 100, "score": 0.80, "image_path": "V1/100.jpg", "timestamp": 10.0}],
                    "label": "text",
                    "weight": 1.0,
                },
                "image_1": {
                    "results": [{"video_id": "V1", "frame_idx": 100, "score": 0.40, "image_path": "V1/100.jpg", "timestamp": 10.0}],
                    "label": "image",
                    "weight": 1.0,
                },
                "ocr_2": {
                    "results": [{"video_id": "V1", "frame_idx": 100, "score": 0.90, "image_path": "V1/100.jpg", "timestamp": 10.0}],
                    "label": "ocr",
                    "weight": 1.0,
                }
            }
        }

        fused = _fuse_engine_results(stages, mock_stage_engine_results, top_k_per_stage=10, rrf_k=60)
        self.assertEqual(len(fused[0]), 1)
        item = fused[0][0]
        self.assertEqual(item.video_id, "V1")
        self.assertEqual(item.frame_idx, 100)
        # Nguồn khớp phải là text_image_fusion và ocr (tổng 2 nguồn RRF, thay vì 3)
        self.assertIn("text_image_fusion", item.matched_sources)
        self.assertIn("ocr", item.matched_sources)
        # RRF của 2 nguồn rank 0: 1.0*(61/61) + 1.0*(61/61) = 2.0
        self.assertAlmostEqual(item.score, 2.0, places=5)

    def test_10_stage_with_image_only_no_fusion(self):
        """
        Test 10: Khi stage chỉ có image_query (không có text) -> image là 1 run độc lập trong RRF bình thường.
        """
        from search.temporal_search import _fuse_engine_results

        stages = [
            StageQuery(
                stage_index=0,
                queries=[
                    StageSubQuery(type="image", query="L22_V031_100", weight=1.0),
                    StageSubQuery(type="ocr", text="VIETNAM", weight=1.0),
                ]
            )
        ]

        mock_stage_engine_results = {
            0: {
                "image_0": {
                    "results": [{"video_id": "V1", "frame_idx": 100, "score": 0.80, "image_path": "V1/100.jpg", "timestamp": 10.0}],
                    "label": "image",
                    "weight": 1.0,
                },
                "ocr_1": {
                    "results": [{"video_id": "V1", "frame_idx": 100, "score": 0.90, "image_path": "V1/100.jpg", "timestamp": 10.0}],
                    "label": "ocr",
                    "weight": 1.0,
                }
            }
        }

        fused = _fuse_engine_results(stages, mock_stage_engine_results, top_k_per_stage=10, rrf_k=60)
        self.assertEqual(len(fused[0]), 1)
        item = fused[0][0]
        self.assertEqual(item.video_id, "V1")
        self.assertEqual(item.frame_idx, 100)
        # Nguồn khớp phải là image và ocr độc lập
        self.assertIn("image", item.matched_sources)
        self.assertIn("ocr", item.matched_sources)
        self.assertNotIn("text_image_fusion", item.matched_sources)

    def test_11_stage_weight_priority(self):
        """
        Test 11: 2 stage, Stage 0 có stage_weight=1.0, Stage 1 có stage_weight=3.0.
        Video A: Stage 0 score=0.90, Stage 1 score=0.20
        Video B: Stage 0 score=0.40, Stage 1 score=0.85
        Weighted avg:
        Video A: (1.0*0.90 + 3.0*0.20)/4.0 = 1.50/4.0 = 0.375
        Video B: (1.0*0.40 + 3.0*0.85)/4.0 = 2.95/4.0 = 0.7375
        Verify: Video B phải được xếp hạng cao hơn Video A.
        """
        from search.temporal_search import _beam_search_dp_alignment

        stages = [
            StageQuery(stage_index=0, text_query="Event 1", stage_weight=1.0),
            StageQuery(stage_index=1, text_query="Event 2", stage_weight=3.0)
        ]

        cands_A = {
            0: [StageEventResult(event_id="Stage 1", stage_index=0, video_id="VideoA", keyframe_id="kA_0", frame_idx=100, timestamp=10.0, image_path="p", score=0.90)],
            1: [StageEventResult(event_id="Stage 2", stage_index=1, video_id="VideoA", keyframe_id="kA_1", frame_idx=200, timestamp=20.0, image_path="p", score=0.20)]
        }
        cands_B = {
            0: [StageEventResult(event_id="Stage 1", stage_index=0, video_id="VideoB", keyframe_id="kB_0", frame_idx=100, timestamp=10.0, image_path="p", score=0.40)],
            1: [StageEventResult(event_id="Stage 2", stage_index=1, video_id="VideoB", keyframe_id="kB_1", frame_idx=200, timestamp=20.0, image_path="p", score=0.85)]
        }

        res_A = _beam_search_dp_alignment("VideoA", stages, cands_A)
        res_B = _beam_search_dp_alignment("VideoB", stages, cands_B)

        self.assertEqual(len(res_A), 1)
        self.assertEqual(len(res_B), 1)

        self.assertAlmostEqual(res_A[0].global_score, 0.375, places=3)
        self.assertAlmostEqual(res_B[0].global_score, 0.7375, places=3)
        self.assertGreater(res_B[0].global_score, res_A[0].global_score)


if __name__ == "__main__":
    unittest.main()
