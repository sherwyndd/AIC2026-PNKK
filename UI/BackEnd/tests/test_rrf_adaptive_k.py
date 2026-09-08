"""Unit tests for RRF configuration and adaptive k values."""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.rrf_config import (
    RRF_K_BY_SOURCE,
    DEFAULT_RRF_K,
    QDRANT_FETCH_MULTIPLIER,
    QDRANT_FETCH_MIN,
)
from search.fusion import reciprocal_rank_fusion


class TestQdrantFetchDepth(unittest.TestCase):
    """Test Qdrant fetch depth calculation."""

    def test_fetch_depth_formula_small_top_k(self):
        """Test fetch depth for small top_k (result should use QDRANT_FETCH_MIN)."""
        top_k = 5
        expected = max(top_k * QDRANT_FETCH_MULTIPLIER, QDRANT_FETCH_MIN)
        self.assertEqual(expected, max(5 * 10, 100))
        self.assertEqual(expected, 100)

    def test_fetch_depth_formula_medium_top_k(self):
        """Test fetch depth for medium top_k."""
        top_k = 15
        expected = max(top_k * QDRANT_FETCH_MULTIPLIER, QDRANT_FETCH_MIN)
        self.assertEqual(expected, max(15 * 10, 100))
        self.assertEqual(expected, 150)

    def test_fetch_depth_formula_large_top_k(self):
        """Test fetch depth for large top_k (multiplier dominates)."""
        top_k = 50
        expected = max(top_k * QDRANT_FETCH_MULTIPLIER, QDRANT_FETCH_MIN)
        self.assertEqual(expected, max(50 * 10, 100))
        self.assertEqual(expected, 500)

    def test_fetch_depth_always_gte_min(self):
        """Test that fetch depth never goes below QDRANT_FETCH_MIN."""
        for top_k in [1, 5, 9]:
            result = max(top_k * QDRANT_FETCH_MULTIPLIER, QDRANT_FETCH_MIN)
            self.assertGreaterEqual(result, QDRANT_FETCH_MIN)


class TestAdaptiveRRFk(unittest.TestCase):
    """Test adaptive RRF k values per source."""

    def test_rrf_k_by_source_structure(self):
        """Test that RRF_K_BY_SOURCE has expected keys and values."""
        required_sources = ["semantic", "asr", "ocr"]
        for source in required_sources:
            self.assertIn(source, RRF_K_BY_SOURCE)
            self.assertIsInstance(RRF_K_BY_SOURCE[source], int)
            self.assertGreater(RRF_K_BY_SOURCE[source], 0)

    def test_rrf_k_value_ranges(self):
        """Test that k values are in expected ranges."""
        # Semantic should have lower k (high confidence)
        self.assertLess(RRF_K_BY_SOURCE["semantic"], 50)

        # Text sources should have higher k (diversity)
        self.assertGreater(RRF_K_BY_SOURCE["asr"], 80)
        self.assertGreater(RRF_K_BY_SOURCE["ocr"], 80)

    def test_rrf_single_source_with_adaptive_k(self):
        """Test RRF with single source uses source_k value correctly."""
        # Create mock results for semantic source
        mock_results = [
            {
                "keyframe_id": "vid1_kf0001",
                "video_id": "vid1",
                "frame_idx": 0,
                "image_path": "vid1/img0.jpg",
                "score": 1.0,
            },
            {
                "keyframe_id": "vid1_kf0002",
                "video_id": "vid1",
                "frame_idx": 30,
                "image_path": "vid1/img1.jpg",
                "score": 0.9,
            },
        ]

        # With source_k, should use k=45 for semantic
        fused = reciprocal_rank_fusion(
            results_by_source={"semantic": mock_results},
            top_k=10,
            k=60,  # global k (not used)
            source_k=RRF_K_BY_SOURCE,
        )

        self.assertEqual(len(fused), 2)
        # Scores should be calculated with k=45 (for semantic)
        # Expected: weight_semantic=2.0, so: 2.0/(45+1) = 0.043478 and 2.0/(45+2) = 0.042553
        k_semantic = RRF_K_BY_SOURCE["semantic"]
        semantic_weight = 2.0  # default weight
        expected_doc1_score = semantic_weight / (k_semantic + 1)
        expected_doc2_score = semantic_weight / (k_semantic + 2)
        
        self.assertAlmostEqual(fused[0]["score"], expected_doc1_score, places=5)
        self.assertAlmostEqual(fused[1]["score"], expected_doc2_score, places=5)

    def test_rrf_multiple_sources_different_k_values(self):
        """Test RRF properly applies different k values for different sources."""
        semantic_results = [
            {
                "keyframe_id": "doc1",
                "video_id": "vid1",
                "frame_idx": 0,
                "image_path": "vid1/img.jpg",
                "score": 1.0,
            }
        ]
        asr_results = [
            {
                "keyframe_id": "doc1",
                "video_id": "vid1",
                "frame_idx": 0,
                "image_path": "vid1/img.jpg",
                "score": 0.8,
            },
            {
                "keyframe_id": "doc2",
                "video_id": "vid2",
                "frame_idx": 100,
                "image_path": "vid2/img.jpg",
                "score": 0.7,
            },
        ]

        fused = reciprocal_rank_fusion(
            results_by_source={
                "semantic": semantic_results,
                "asr": asr_results,
            },
            top_k=10,
            k=60,  # fallback
            source_k=RRF_K_BY_SOURCE,
        )

        # doc1 should score: w_semantic/(45+1) + w_asr/(90+1)
        # doc2 should score: w_asr/(90+2)
        # Since semantic weight=2.0, asr weight=1.0
        # doc1 = 2.0/(45+1) + 1.0/(90+1) = 0.04348 + 0.01099 = 0.05447
        # doc2 = 1.0/(90+2) = 0.01087

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["keyframe_id"], "doc1")
        self.assertGreater(fused[0]["score"], fused[1]["score"])
        self.assertEqual(fused[1]["keyframe_id"], "doc2")

    def test_rrf_falls_back_to_global_k(self):
        """Test RRF falls back to global k for sources not in source_k."""
        results = [
            {
                "keyframe_id": "doc1",
                "video_id": "vid1",
                "frame_idx": 0,
                "image_path": "vid1/img.jpg",
                "score": 1.0,
            }
        ]

        # Use custom source_k without "custom_source"
        custom_k = {"semantic": 40}

        fused = reciprocal_rank_fusion(
            results_by_source={"custom_source": results},
            top_k=10,
            k=60,  # global fallback
            source_k=custom_k,
        )

        # Should use k=60 for custom_source (not in custom_k)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["score"], round(1.0 / (60 + 1), 6), places=5)

    def test_rrf_many_sources_with_different_k(self):
        """Test RRF with all 4 sources using different k values."""
        results_by_source = {
            "semantic": [
                {
                    "keyframe_id": "doc1",
                    "video_id": "vid1",
                    "frame_idx": 0,
                    "image_path": "vid1/img.jpg",
                    "score": 1.0,
                }
            ],
            "asr": [
                {
                    "keyframe_id": "doc1",
                    "video_id": "vid1",
                    "frame_idx": 0,
                    "image_path": "vid1/img.jpg",
                    "score": 0.8,
                },
                {
                    "keyframe_id": "doc2",
                    "video_id": "vid2",
                    "frame_idx": 100,
                    "image_path": "vid2/img.jpg",
                    "score": 0.7,
                },
            ],
            "ocr": [
                {
                    "keyframe_id": "doc1",
                    "video_id": "vid1",
                    "frame_idx": 0,
                    "image_path": "vid1/img.jpg",
                    "score": 0.9,
                },
                {
                    "keyframe_id": "doc3",
                    "video_id": "vid3",
                    "frame_idx": 50,
                    "image_path": "vid3/img.jpg",
                    "score": 0.6,
                },
            ],
            "object": [
                {
                    "keyframe_id": "doc1",
                    "video_id": "vid1",
                    "frame_idx": 0,
                    "image_path": "vid1/img.jpg",
                    "score": 0.85,
                },
            ],
        }

        fused = reciprocal_rank_fusion(
            results_by_source=results_by_source,
            top_k=10,
            k=60,
            source_k=RRF_K_BY_SOURCE,
        )

        # doc1 should be top (appears in all 4 sources)
        self.assertEqual(fused[0]["keyframe_id"], "doc1")

        # doc2 and doc3 should follow (appear in 2 sources each)
        remaining_ids = [fused[i]["keyframe_id"] for i in range(1, len(fused))]
        self.assertIn("doc2", remaining_ids)
        self.assertIn("doc3", remaining_ids)

        # Verify all docs are present
        fused_ids = {item["keyframe_id"] for item in fused}
        self.assertEqual(fused_ids, {"doc1", "doc2", "doc3"})


class TestConfigurationConsistency(unittest.TestCase):
    """Test configuration consistency and correctness."""

    def test_default_rrf_k_is_defined(self):
        """Test that DEFAULT_RRF_K is defined and positive."""
        self.assertIsNotNone(DEFAULT_RRF_K)
        self.assertGreater(DEFAULT_RRF_K, 0)
        self.assertEqual(DEFAULT_RRF_K, 60)

    def test_qdrant_config_constants(self):
        """Test Qdrant configuration constants are sensible."""
        self.assertEqual(QDRANT_FETCH_MULTIPLIER, 10)
        self.assertEqual(QDRANT_FETCH_MIN, 100)
        self.assertGreater(QDRANT_FETCH_MULTIPLIER, 1)
        self.assertGreater(QDRANT_FETCH_MIN, 10)


if __name__ == "__main__":
    unittest.main()
