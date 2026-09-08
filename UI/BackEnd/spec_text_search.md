# SPEC: Tích hợp Search OCR + ASR (Elasticsearch/BM25) vào hệ thống search hiện có

## 1. Hạ tầng / Kết nối

| Thành phần | Giá trị |
|---|---|
| Elasticsearch host | `http://127.0.0.1:9200` |
| Elasticsearch index | `aic2026_elastics_text` |
| Qdrant host | `http://127.0.0.1:6333` |
| Qdrant collection (vector ảnh) | `image_siglip` |

> Toàn bộ query/API trong spec này mặc định trỏ tới `aic2026_elastics_text` trên host `http://127.0.0.1:9200`, trừ khi ghi chú khác.

---

## 2. Bối cảnh

Hệ thống hiện tại đã có:
- **Search theo hình ảnh/semantic** qua Qdrant (`image_siglip` collection, SigLIP2 embedding, 768-dim, cosine distance)
- **Elasticsearch index** `aic2026_elastics_text` (274,542 document) với các field:
  `asr_text`, `caption_text`, `dense_ocr_text`, `object_text`, `ocr_text`, `merged_text`,
  cùng metadata: `video_id`, `keyframe_id`, `frame_idx`, `image_path`

Trên frontend đã có sẵn 2 ô search **OCR** và **ASR/Text** nhưng chưa nối logic backend — hiện đang bỏ trống, không gọi API nào.

### Mục tiêu của spec này
1. Kết nối 2 ô search có sẵn trên frontend (**OCR**, **ASR**) vào Elasticsearch, dùng BM25.
2. Khi người dùng nhập **nhiều điều kiện search cùng lúc** (ví dụ: OCR + ASR, hoặc OCR + Text ảnh Qdrant), hệ thống fuse các nguồn kết quả bằng **Reciprocal Rank Fusion (RRF)**, trả về danh sách top-K cuối cùng đã gộp.

---

## 3. Phạm vi (Scope)

### Trong phạm vi
- Search OCR (field `ocr_text`, dùng BM25 + fuzzy, config đã thống nhất — xem mục 5)
- Search ASR (field `asr_text`, dùng BM25 — xem mục 6)
- RRF fusion khi ≥2 nguồn search được active cùng lúc (OCR, ASR, và/hoặc Qdrant vector search hiện có)
- API backend mới/mở rộng để frontend gọi

### Ngoài phạm vi (không làm ở spec này)
- Search `caption_text`, `object_text`, `dense_ocr_text`, `merged_text` (để lại cho spec sau nếu cần)
- Custom Vietnamese analyzer (đã quyết định dùng `standard` analyzer có sẵn — xem Quyết định #1)
- Native Qdrant group-by (đang bị lỗi timeout, xử lý ở task riêng)

---

## 4. Quyết định thiết kế đã chốt

| # | Quyết định | Lý do |
|---|---|---|
| 1 | Dùng `standard` analyzer có sẵn cho `ocr_text` và `asr_text`, **không** làm custom Vietnamese analyzer | `standard` đã xử lý Unicode/dấu tiếng Việt ổn, đủ dùng cho text ngắn (OCR) và transcript (ASR); tránh downtime/reindex không cần thiết |
| 2 | OCR chỉ search trên field `ocr_text` (không dùng `dense_ocr_text`) | Đã chốt theo yêu cầu — `ocr_text` là nguồn OCR chính, sạch hơn |
| 3 | Query luôn có dấu tiếng Việt đầy đủ | Không cần xử lý bỏ dấu/so khớp không dấu ở version này |
| 4 | Có fuzzy matching cho OCR (không áp dụng fuzzy cho ASR) | OCR dễ bị lỗi đọc sai ký tự (ảnh mờ/nghiêng); ASR transcript thường ít lỗi chính tả hơn, ưu tiên match chính xác hơn để tránh nhiễu |
| 5 | Kết hợp đa nguồn bằng RRF (không dùng linear score combination) | RRF không phụ thuộc vào thang điểm khác nhau giữa BM25 (ES) và Cosine similarity (Qdrant) — công bằng hơn khi merge |

---

## 5. Search OCR — Config chi tiết

### Field: `ocr_text` (type: `text`, analyzer: `standard`)

### Query template

```json
{
  "query": {
    "match": {
      "ocr_text": {
        "query": "<user_input>",
        "fuzziness": "AUTO",
        "prefix_length": 1,
        "operator": "or",
        "minimum_should_match": "70%"
      }
    }
  },
  "size": "<fetch_depth>",
  "min_score": 1.0,
  "_source": ["video_id", "keyframe_id", "frame_idx", "image_path", "ocr_text"]
}
```

### Giải thích tham số

| Tham số | Giá trị | Lý do |
|---|---|---|
| `fuzziness` | `AUTO` | Tự điều chỉnh số lỗi cho phép theo độ dài từ, phù hợp với OCR text đa dạng độ dài |
| `prefix_length` | `1` | Giữ ký tự đầu chính xác, giảm false positive từ fuzzy match lan man |
| `operator` | `or` | Không bắt buộc match hết mọi từ (OCR ngắn, dễ mất 1 từ do lỗi đọc) |
| `minimum_should_match` | `70%` | Cân bằng — không quá lỏng (or thuần) không quá chặt (and thuần) |
| `min_score` | `1.0` | Lọc bớt kết quả fuzzy match quá xa, điểm thấp — **cần tune lại sau khi test thật với data** |

### API endpoint

```
POST /api/search/ocr
Body: {
  "query": "Vinamilk",
  "top_k": 20,
  "video_filter": null   // optional, filter theo video_id cụ thể
}
```

### Response

```json
{
  "source": "ocr",
  "results": [
    {
      "score": 12.4,
      "video_id": "L21_V001",
      "keyframe_id": "L21_V001_kf_0844",
      "frame_idx": 844,
      "image_path": "...",
      "ocr_text": "Vinamilk 100% Fresh Milk"
    }
  ]
}
```

---

## 6. Search ASR — Config chi tiết

### Field: `asr_text` (type: `text`, analyzer: `standard`)

### Query template

```json
{
  "query": {
    "match": {
      "asr_text": {
        "query": "<user_input>",
        "operator": "or",
        "minimum_should_match": "60%"
      }
    }
  },
  "size": "<fetch_depth>",
  "min_score": 1.0,
  "_source": ["video_id", "keyframe_id", "frame_idx", "image_path", "asr_text"]
}
```

### Giải thích tham số

| Tham số | Giá trị | Lý do |
|---|---|---|
| `fuzziness` | *(không dùng)* | Transcript ASR thường ít lỗi ký tự đơn lẻ hơn OCR; fuzzy dễ gây nhiễu trên văn bản dài |
| `operator` | `or` | ASR transcript dài, câu query có thể chỉ trùng 1 phần cụm từ |
| `minimum_should_match` | `60%` | Nới hơn OCR (70%) vì transcript dài hơn, tỷ lệ match tự nhiên thấp hơn dù đúng ngữ cảnh |
| `min_score` | `1.0` | Baseline, cần tune lại theo test thật |

> **Ghi chú:** giá trị `minimum_should_match` và `min_score` cho ASR là đề xuất ban đầu, cần A/B test với data thật trước khi chốt version production (xem mục 9 — Việc cần làm khi test).

### API endpoint

```
POST /api/search/asr
Body: {
  "query": "xin chào các bạn",
  "top_k": 20,
  "video_filter": null
}
```

### Response

Cấu trúc giống mục 5, `"source": "asr"`.

---

## 7. RRF Fusion — khi search nhiều nguồn cùng lúc

### 6.1 Khi nào kích hoạt RRF

Khi người dùng nhập **từ 2 ô search trở lên cùng lúc** (bất kỳ tổ hợp nào trong: OCR, ASR, Qdrant vector/semantic search hiện có). Nếu chỉ 1 ô có input → trả kết quả trực tiếp từ nguồn đó, không cần fusion.

### 6.2 Công thức RRF

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i ∈ các nguồn có chứa doc
```

- `rank_i(doc)`: thứ hạng (1-based) của `doc` trong danh sách kết quả trả về từ nguồn `i`
- `k`: hằng số làm mượt (damping), **mặc định `k = 60`** (đã dùng nhất quán với `rrf_k=60` hiện có trong log backend hiện tại)
- Nếu 1 document không xuất hiện trong 1 nguồn nào đó → nguồn đó đóng góp `0` vào tổng (không tính)

### 6.3 Định danh document để merge (join key)

Vì Qdrant và Elasticsearch là 2 hệ thống riêng biệt, cần 1 khóa chung để biết "đây là cùng 1 keyframe":

```
join_key = keyframe_id   (ưu tiên, có sẵn ở cả 2 nguồn)
```

Nếu `keyframe_id` không đồng nhất giữa 2 hệ thống (cần xác nhận — xem mục 9, câu hỏi mở), dùng tổ hợp `(video_id, frame_idx)` làm fallback.

### 6.4 Luồng xử lý

```
1. Với mỗi nguồn active (OCR / ASR / Qdrant):
     fetch_depth = top_k × 3   (áp dụng multiplier đã thống nhất trước đó)
     results_i = search_nguồn_i(query_i, limit=fetch_depth)

2. Gán rank cho từng doc trong mỗi results_i (rank bắt đầu từ 1)

3. Với mỗi doc xuất hiện ở ít nhất 1 nguồn:
     rrf_score(doc) = Σ 1/(k + rank_i(doc))  qua các nguồn có chứa doc

4. Sort toàn bộ theo rrf_score giảm dần

5. Trả về top_k đầu tiên sau khi sort
```

### 6.5 Pseudo-code

```python
def reciprocal_rank_fusion(results_by_source: dict[str, list[dict]], top_k: int, k: int = 60):
    """
    results_by_source = {
        "ocr":    [{"keyframe_id": "...", ...}, ...],   # đã sort theo score giảm dần
        "asr":    [{"keyframe_id": "...", ...}, ...],
        "qdrant": [{"keyframe_id": "...", ...}, ...],
    }
    """
    scores = {}       # keyframe_id -> rrf_score
    doc_data = {}      # keyframe_id -> full doc info (giữ lại metadata)

    for source_name, results in results_by_source.items():
        for rank, doc in enumerate(results, start=1):
            key = doc["keyframe_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_data:
                doc_data[key] = doc
            doc_data[key].setdefault("matched_sources", []).append(source_name)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final = []
    for key, score in ranked[:top_k]:
        d = doc_data[key]
        d["rrf_score"] = score
        final.append(d)
    return final
```

### 6.6 API endpoint tổng hợp (multi-source search)

```
POST /api/search
Body: {
  "ocr_query": "Vinamilk",          // optional
  "asr_query": "xin chào các bạn",   // optional
  "semantic_query": "a man eating",  // optional, đi qua Qdrant/SigLIP hiện có
  "top_k": 20,
  "rrf_k": 60,
  "video_filter": null
}
```

Logic backend:
- Chỉ gọi search cho nguồn nào có `query` khác `null`/rỗng
- Nếu chỉ 1 nguồn active → bỏ qua RRF, trả thẳng kết quả nguồn đó (giữ nguyên `score` gốc, không cần `rrf_score`)
- Nếu ≥2 nguồn active → chạy RRF theo mục 7.4-7.5

### Response

```json
{
  "sources_used": ["ocr", "asr"],
  "results": [
    {
      "keyframe_id": "L21_V001_kf_0844",
      "video_id": "L21_V001",
      "frame_idx": 844,
      "image_path": "...",
      "rrf_score": 0.0328,
      "matched_sources": ["ocr", "asr"],
      "ocr_text": "...",
      "asr_text": "..."
    }
  ]
}
```

---

## 8. Frontend — việc cần nối

| UI hiện có | Cần làm |
|---|---|
| Ô search **OCR** (đang bỏ trống) | Bind vào `ocr_query` trong request body `/api/search` |
| Ô search **ASR** (đang bỏ trống) | Bind vào `asr_query` trong request body `/api/search` |
| Ô search **semantic/hình ảnh** (đã hoạt động) | Giữ nguyên, chỉ đổi tên field gửi lên thành `semantic_query` để khớp API mới |
| Hiển thị kết quả | Thêm badge/label nhỏ trên mỗi thumbnail cho biết kết quả match từ nguồn nào (`matched_sources`) — hữu ích để user biết vì sao ảnh đó xuất hiện |

---

## 9. Việc cần làm khi test / câu hỏi còn mở

1. **Xác nhận `keyframe_id` có đồng nhất giữa Qdrant và Elasticsearch không** (cùng format, cùng giá trị cho cùng 1 keyframe) — quyết định join key chính xác cho RRF.
2. **Tune `min_score` và `minimum_should_match`** cho cả OCR và ASR bằng test thật trên data (giá trị trong spec là baseline ban đầu, chưa phải final).
3. **Đo latency riêng từng nguồn** (OCR-only, ASR-only, RRF 2-3 nguồn) để biết fetch_depth = top_k × 3 có phù hợp không, hay cần điều chỉnh theo từng nguồn riêng (ví dụ ES thường nhanh hơn Qdrant nhiều, có thể fetch_depth ES để cao hơn không ảnh hưởng latency tổng).
4. **Xử lý trường hợp 1 nguồn không đủ fetch_depth kết quả** (ví dụ OCR chỉ ra 5 kết quả dù fetch_depth=60) — RRF vẫn hoạt động bình thường (nguồn đó đóng góp ít hơn), không cần xử lý đặc biệt, nhưng cần log lại để theo dõi.
5. **Weight khác nhau giữa các nguồn?** — hiện spec dùng RRF thuần (mọi nguồn đóng góp ngang nhau theo rank). Nếu sau này muốn ưu tiên 1 nguồn hơn (ví dụ OCR đáng tin hơn ASR), có thể mở rộng công thức thành weighted RRF:
   ```
   RRF_score(doc) = Σ  w_i / (k + rank_i(doc))
   ```
   — để ngoài phạm vi version 1 này, ghi nhận làm hướng mở rộng.

---

## 10. Tóm tắt việc cần code

- [ ] `search/ocr_search.py` — hàm `search_ocr(query, top_k, video_filter)`
- [ ] `search/asr_search.py` — hàm `search_asr(query, top_k, video_filter)`
- [ ] `search/fusion.py` — hàm `reciprocal_rank_fusion(results_by_source, top_k, k=60)`
- [ ] Sửa `main.py` (hoặc router search hiện có) — mở rộng `/api/search` nhận `ocr_query`, `asr_query`, `semantic_query`, điều phối gọi đúng nguồn và fusion
- [ ] Frontend — bind 2 ô OCR/ASR có sẵn vào request body mới, thêm hiển thị `matched_sources`
- [ ] Log latency riêng từng nguồn + tổng, theo format nhất quán với log hiện có (`SEARCH | query=... top_k=... fetch_depth=...`).