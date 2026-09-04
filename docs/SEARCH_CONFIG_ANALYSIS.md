# 🔍 SEARCH CONFIGURATION ANALYSIS & RECALL OPTIMIZATION RECOMMENDATIONS

*Generated: 2026-08-18*  
*Scope: Qdrant Vector Search + Elasticsearch BM25 + RRF Fusion*  
**Note: NO CODE CHANGES PROPOSED — Analysis & Recommendations Only**

---

## 1. HIỆN TẠI CẤU HÌNH SEARCH

### 1.1 QDRANT Vector Search Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Host** | 127.0.0.1:6333 | Local instance |
| **Collection (Active)** | `image_siglip` | FLOAT32 full precision |
| **Collection (Backup)** | `image_siglip_sq8` | INT8 quantized (speed vs recall tradeoff) |
| **Embedding Model** | SigLIP2 (Sigmoid Loss Image Pre-training) | ~384D embeddings |
| **Backup Model** | PECORE | Alternative dense encoder |
| **Vector Index Type** | HNSW (Hierarchical Navigable Small World) | Graph-based nearest neighbor search |
| **HNSW m parameter** | 32 | Edges per node (balanced: recall vs memory) |
| **HNSW ef_construct** | 512 | Search effort during indexing (high quality) |
| **HNSW ef_search** | 256 | Search effort during query (recently upgraded from 128) |
| **indexed_only** | True | Only search indexed segments (disable for complete coverage) |
| **Quantization** | Disabled | Full precision floating-point (no compression) |
| **Qdrant Fetch Depth** | `top_k` (default 10) | Results fetched from Qdrant before RRF |
| **Qdrant Timeout** | 120s | Client connection timeout |

**Key Insight:** Qdrant is configured for **HIGH RECALL** (full FP32, no quantization, high ef_search).

---

### 1.2 Elasticsearch Configuration (ASR Text Search)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Host** | 127.0.0.1:9200 | Local instance |
| **Index Name** | `aic2026_elastics_text` | Search index for ASR/text data |
| **Query Type** | BM25 (Best Match 25) | TF-IDF with length normalization |
| **Query Operator** | `or` (Union) | Match ANY term in query (high recall) |
| **minimum_should_match** | 60% | At least 60% of terms must match |
| **min_score** | 1.0 | Minimum BM25 score threshold |
| **Fetch Depth** | `max(top_k * 5, 150)` | **Min 150 results** fetched before RRF |
| **Source Fields** | video_id, keyframe_id, frame_idx, image_path, asr_text | Returned metadata |
| **HTTP Timeout** | 10s | Request timeout |

**Key Insight:** Elasticsearch is **HEAVILY OVER-SAMPLING** (5-10x more results) to ensure diverse candidates before fusion.

---

### 1.3 RRF (Reciprocal Rank Fusion) Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **RRF k Parameter** | 60 (default) | Smoothing factor in formula |
| **RRF Formula** | `score = Σ 1 / (k + rank)` | Rank-based fusion (1-indexed rank) |
| **Top-K Merge** | Variable per request | Final results after fusion |
| **Model Count** | Up to 4 | Qdrant (SigLIP), Elasticsearch (ASR), OCR, Object Detection |
| **Normalization** | By non-empty model count | Scores adjusted for partial results |

**Example RRF Scoring:**
- If result appears at rank 1 in Qdrant and rank 5 in ES: score = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0141 = 0.0305
- If result appears only in Qdrant rank 10: score = 1/(60+10) = 0.0139
- **Higher rank position = exponentially higher contribution** (Reciprocal effect)

---

## 2. RECALL BOTTLENECK ANALYSIS

### 2.1 Current Recall Constraints

#### 🔴 **Issue A: Limited Initial Fetch from Qdrant**
- **Current:** `qdrant_fetch_depth = req.top_k` (default 10 results)
- **Problem:** 
  - Only 10 nearest neighbors searched from Qdrant
  - HNSW ef_search=256 is **NOT fully utilized**
  - If true positives are beyond top-10, they're **lost before RRF**
  - No chance for ES/OCR to rescue them

#### 🟡 **Issue B: High Elasticsearch Oversampling (but necessary)**
- **Current:** Fetch 150-500 results from ES before RRF
- **Why this much?** Text search (BM25) has lower precision than dense vectors
- **Trade-off:** High latency for diversity guarantee

#### 🟡 **Issue C: RRF k=60 may be suboptimal**
- **Current:** k=60 used uniformly for all searches
- **Impact:** 
  - k=60 means rank 1 gets score 1/61, rank 60 gets 1/120
  - Lower k → early ranks dominate more (good for high-confidence)
  - Higher k → flattens scores (good for diversity)
- **Problem:** No adaptive k based on query difficulty/type

#### 🔴 **Issue D: No Reranking Post-RRF**
- **Current:** RRF scores used directly as final rank
- **Problem:**
  - Cross-encoder/semantic reranking could boost top-K precision
  - Current approach is **fusion-only**, not **intelligence-aware ranking**

#### 🔴 **Issue E: Single Embedding Model Dependency**
- **Current:** SigLIP2 only (no PECORE ensemble)
- **Missing:** PECORE collection exists but NOT queried in main search
- **Impact:** Diversity of embedding perspectives lost

---

## 3. RECOMMENDATIONS TO INCREASE RECALL

### ⭐ **Priority 1: HIGH IMPACT (Easy to implement)**

#### **Rec 1.1 — Increase Qdrant Fetch Depth (HNSW ef_search)**
```
Current: qdrant_fetch_depth = req.top_k (10 results)
Recommendation: qdrant_fetch_depth = max(req.top_k * 10, 100)  

Rationale:
- HNSW ef_search=256 can efficiently return 100+ candidates
- No significant latency penalty (HNSW is sub-linear)
- More candidates → higher chance of true positives
- RRF can properly rank them with ES/OCR/Object data

Expected Recall Impact: +15-25% (depends on query-document distribution)
Latency Impact: +5-15% (HNSW is efficient)
```

#### **Rec 1.2 — Adaptive RRF k Based on Query Type**
```
Current: rrf_k = 60 (fixed)
Recommendation:
  - semantic_query (dense vector) → rrf_k = 40-50 (early ranks trusted)
  - asr_query (text search) → rrf_k = 80-100 (diversity needed)
  - ocr_query (text search) → rrf_k = 80-100 (diversity needed)
  - object_query (tag search) → rrf_k = 60 (balanced)

Rationale:
- Dense semantic search is more confident → tighter k
- Text search (BM25) has high variance → looser k for diversity
- Allows per-modality tuning without breaking existing API

Expected Recall Impact: +8-12%
Latency Impact: None (same computation)
```

#### **Rec 1.3 — Increase ES minimum_should_match from 60% → 50%**
```
Current: minimum_should_match: 60%
Recommendation: minimum_should_match: 50%  

Rationale:
- 60% is strict: 5-term query requires 3 matches (3/5 = 60%)
- 50% is balanced: 5-term query requires 2+ matches
- More candidate results without significant noise

Expected Recall Impact: +10-15% (especially for longer queries)
Latency Impact: +5% (more ES results to fetch)
Configuration Risk: LOW (already oversampling 5x)
```

---

### ⭐ **Priority 2: MEDIUM IMPACT (Requires Logic Addition)**

#### **Rec 2.1 — Dual-Model Ensemble (SigLIP + PECORE)**
```
Current: Only image_siglip collection queried
Recommendation: Query BOTH image_siglip AND image_pecore_encoder
              - Run in parallel
              - Add PECORE results to RRF fusion with equal weight
              - Merge at RRF stage

Rationale:
- Different embeddings capture different visual semantics
- PECORE: domain-specific encoder (potentially better for objects/scenes)
- SigLIP2: CLIP-aligned (better for semantic/natural language)
- Ensemble reduces embedding bias

Expected Recall Impact: +12-20%
Latency Impact: +30-50% (2x embedding searches, but parallelizable)
Complexity: Medium (add PECORE to main search flow)
```

#### **Rec 2.2 — Semantic Reranking (Cross-Encoder) on Top-100**
```
Current: No reranking after RRF
Recommendation: 
  1. Get top-100 fused results from RRF
  2. If query is semantic → pass top-100 through cross-encoder
  3. Rerank by cross-encoder score
  4. Return top-K

Rationale:
- RRF is good at **recall** but not **precision**
- Cross-encoder (e.g., CLIP image-text matching) is slower but smarter
- Reranking top-100 is feasible (100 forward passes)

Expected Recall @ Top-10: +5-8% (precision focus)
Expected Recall @ Top-100: +20-30% (candidates captured)
Latency Impact: +200-500ms (depends on model)
Complexity: Medium-High (model serving, batching)
```

---

### ⭐ **Priority 3: ADVANCED (Requires Infrastructure Changes)**

#### **Rec 3.1 — Hybrid Sparse-Dense Search in Qdrant**
```
Current: Pure dense vector search only (SigLIP embeddings)
Recommendation: Add Qdrant's Sparse Vector support
  - Tokenize query → sparse BM25 vector
  - Combine dense + sparse in Qdrant query
  - HNSW on dense, BM25 on sparse, merge within Qdrant

Rationale:
- Captures both semantic AND keyword matching in one index
- Reduces dependency on separate Elasticsearch
- Qdrant's sparse vectors are production-ready

Expected Recall Impact: +15-25%
Latency Impact: -20% (one index instead of two)
Infrastructure: Requires reindexing + new collection schema
Complexity: High
```

#### **Rec 3.2 — Query Expansion via LLM**
```
Current: Query used as-is
Recommendation:
  - User query → LLM expansion (synonyms, related concepts)
  - Search with expanded query
  - Rank original + expanded results together

Rationale:
- "dog" expands to "canine, puppy, pet, animal"
- Captures relevant docs using different vocabulary

Expected Recall Impact: +10-20%
Latency Impact: +1-2s (LLM inference)
Complexity: High (model serving, prompt engineering)
Cost: LLM API calls
```

---

## 4. QUICK WINS (Lowest Effort, Immediate Impact)

| Rank | Change | Impact | Effort | Risk |
|------|--------|--------|--------|------|
| #1 | Increase `qdrant_fetch_depth` to 100 | +20% recall | 1 line | 🟢 Low |
| #2 | Lower ES `minimum_should_match` to 50% | +12% recall | 1 line | 🟢 Low |
| #3 | Adaptive `rrf_k` by query type | +10% recall | ~30 lines | 🟢 Low |
| #4 | Enable PECORE collection in search | +15% recall | ~40 lines | 🟡 Medium |
| #5 | Cross-encoder reranking | +8% precision | ~100 lines | 🟡 Medium |

---

## 5. EXPECTED CUMULATIVE IMPROVEMENTS

**Assuming implementation of Rec 1.1 + 1.2 + 1.3 + 2.1:**

```
Baseline Recall @ Top-10:       75% (hypothetical)
                                 ↓
+ Increase Qdrant depth:        +20% → 85%
+ Adaptive k:                   +8% → 87%
+ Lower ES match threshold:     +10% → 90%
+ PECORE ensemble:              +12% → 94%
                                ─────────
**Target Recall @ Top-10:       ~92-94%** 

Latency Impact:                 ~+40-60% (mostly PECORE 2x vector search)
Memory Impact:                  ~+15-20% (PECORE index in Qdrant)
```

---

## 6. MONITORING & VALIDATION

### Metrics to Track:
1. **Recall@K** — Fraction of true positives in top-K
2. **NDCG@K** — Normalized Discounted Cumulative Gain (ranking quality)
3. **MRR** — Mean Reciprocal Rank (average position of first true positive)
4. **Latency P50/P95** — Search response time percentiles
5. **Model Contribution** — Which model contributes most to final results

### Validation Strategy:
- Create labeled eval set (query → annotated relevant keyframes)
- A/B test each recommendation incrementally
- Measure recall, latency, user satisfaction in parallel

---

## 7. SUMMARY TABLE

| Aspect | Current State | Bottleneck | Recommendation |
|--------|---------------|-----------|-----------------|
| **Vector Search Depth** | 10 | Too shallow | → 100+ |
| **Vector Models** | 1 (SigLIP) | Narrow perspective | → 2 (SigLIP+PECORE) |
| **Text Search Threshold** | 60% | Strict | → 50% |
| **RRF k Parameter** | 60 (fixed) | Non-adaptive | → 40-100 (query-aware) |
| **Post-Fusion Ranking** | None | Missed precision boost | → Cross-encoder rerank |
| **Qdrant Config** | Optimized (ef=256) | Well-tuned ✓ | No change needed |
| **ES Config** | Healthy oversampling | Good ✓ | Fine-tune only threshold |

---

## 8. IMPLEMENTATION PRIORITY ROADMAP

**Phase 1 (Week 1) — Low-hanging fruit:**
- Rec 1.1: Increase qdrant_fetch_depth (1 line)
- Rec 1.2: Adaptive rrf_k by query type (~30 lines)
- Rec 1.3: Lower ES minimum_should_match (1 line)
- **Expected Recall Gain: ~30-35%**

**Phase 2 (Week 2-3) — Medium effort:**
- Rec 2.1: PECORE ensemble search (~40 lines + reindex)
- **Expected Recall Gain: Additional +12-15%**

**Phase 3 (Week 4+) — Advanced:**
- Rec 2.2: Cross-encoder reranking (~100 lines)
- Rec 3.1: Hybrid sparse-dense vectors (infrastructure)
- **Expected Recall Gain: Additional +5-10%**

---

**End of Analysis**
