"""RRF (Reciprocal Rank Fusion) configuration with per-source k values."""

# Per-source k parameters for RRF
# Lower k = early ranks contribute more (used for high-confidence sources)
# Higher k = flatter distribution (used for diverse/lower-confidence sources)
RRF_K_BY_SOURCE = {
    "semantic": 45,    # Dense vector search - high confidence
    "asr": 90,         # Text search - lower precision, needs diversity
    "ocr": 90,         # Text search - lower precision, needs diversity
    "object": 60,      # Balanced
}

# Default k if source not specified
DEFAULT_RRF_K = 60

# Qdrant fetch depth configuration
# Default: fetch 10x the requested top_k, minimum 100 results
QDRANT_FETCH_MULTIPLIER = 10
QDRANT_FETCH_MIN = 100
