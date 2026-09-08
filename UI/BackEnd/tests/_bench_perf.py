"""Benchmark search performance bottlenecks."""
import sys, os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.insert(0, "/AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd")

print("=" * 60)
print("STEP 1: Load SigLIP2 model")
t0 = time.time()
from models.siglip_encoder import load_siglip, encode_text_siglip
load_siglip(device="cuda:0")
print(f"  Load time: {time.time() - t0:.2f}s")

encode_text_siglip("warmup")

N = 5
queries = [
    "a person riding a motorcycle on a busy street",
    "two people talking on the street",
    "a tall building in the city center",
    "car driving on highway at night",
    "sunset over the ocean with waves",
]
t0 = time.time()
for q in queries:
    vec = encode_text_siglip(q)
t_enc = (time.time() - t0) / N
print(f"\nSTEP 2: Text encode (avg of {N}): {t_enc*1000:.1f}ms  dim={vec.shape[0]}")

print("\nSTEP 3: Qdrant collection info")
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333, timeout=30)
info = client.get_collection("image_siglip")
print(f"  points_count: {info.points_count}")
print(f"  vectors size: {info.config.params.vectors.size}")
print(f"  distance: {info.config.params.vectors.distance}")
try:
    print(f"  on_disk: {info.config.params.vectors.on_disk}")
except Exception:
    pass
print(f"  hnsw_config: {info.config.hnsw_config}")
print(f"  optimizer_config: {info.config.optimizer_config}")
print(f"  segments_count: {info.segments_count}")

test_vec = encode_text_siglip("a person walking a dog in the park")

print("\nSTEP 4: Qdrant search, limit=100 (no filter)")
N = 3
times = []
for i in range(N):
    t0 = time.time()
    r = client.query_points(collection_name="image_siglip", query=test_vec.tolist(), limit=100)
    t = time.time() - t0
    times.append(t)
    print(f"  run #{i+1}: {t*1000:.0f}ms, hits={len(r.points)}, top={r.points[0].score:.4f}")
print(f"  AVG: {sum(times)/len(times)*1000:.0f}ms")

print("\nSTEP 5: Qdrant search, limit=500 (no filter)")
times = []
for i in range(N):
    t0 = time.time()
    r = client.query_points(collection_name="image_siglip", query=test_vec.tolist(), limit=500)
    t = time.time() - t0
    times.append(t)
    print(f"  run #{i+1}: {t*1000:.0f}ms, hits={len(r.points)}")
print(f"  AVG: {sum(times)/len(times)*1000:.0f}ms")

from qdrant_client.models import SearchParams
print("\nSTEP 6: Qdrant search, limit=100, hnsw_ef=128 (default ~?)")
times = []
for i in range(N):
    t0 = time.time()
    r = client.query_points(
        collection_name="image_siglip", query=test_vec.tolist(), limit=100,
        search_params=SearchParams(hnsw_ef=128)
    )
    t = time.time() - t0
    times.append(t)
    print(f"  run #{i+1}: {t*1000:.0f}ms, top={r.points[0].score:.4f}")
print(f"  AVG: {sum(times)/len(times)*1000:.0f}ms")

print("\nSTEP 7: Qdrant search, limit=100, hnsw_ef=64")
times = []
for i in range(N):
    t0 = time.time()
    r = client.query_points(
        collection_name="image_siglip", query=test_vec.tolist(), limit=100,
        search_params=SearchParams(hnsw_ef=64)
    )
    t = time.time() - t0
    times.append(t)
    print(f"  run #{i+1}: {t*1000:.0f}ms, top={r.points[0].score:.4f}")
print(f"  AVG: {sum(times)/len(times)*1000:.0f}ms")

print("\nSTEP 8: Qdrant search, limit=100, hnsw_ef=32")
times = []
for i in range(N):
    t0 = time.time()
    r = client.query_points(
        collection_name="image_siglip", query=test_vec.tolist(), limit=100,
        search_params=SearchParams(hnsw_ef=32)
    )
    t = time.time() - t0
    times.append(t)
    print(f"  run #{i+1}: {t*1000:.0f}ms, top={r.points[0].score:.4f}")
print(f"  AVG: {sum(times)/len(times)*1000:.0f}ms")

print("\nSTEP 9: Qdrant search, limit=100, exact (ANN=false)")
try:
    times = []
    for i in range(N):
        t0 = time.time()
        r = client.query_points(
            collection_name="image_siglip", query=test_vec.tolist(), limit=100,
            search_params=SearchParams(exact=True)
        )
        t = time.time() - t0
        times.append(t)
        print(f"  run #{i+1}: {t*1000:.0f}ms, top={r.points[0].score:.4f}")
    print(f"  AVG: {sum(times)/len(times)*1000:.0f}ms")
except Exception as e:
    print(f"  exact search not supported: {e}")

print("\nSTEP 10: End-to-end simulate /api/search (encode + query + RRF)")
sys.path.insert(0, ".")
from search.rrf import rrf_fuse
N = 3
times = []
for i in range(N):
    t0 = time.time()
    v = encode_text_siglip("a red car parked near a tree")
    t_enc_here = time.time() - t0
    r = client.query_points(collection_name="image_siglip", query=v.tolist(), limit=500)
    t_q = time.time() - t0 - t_enc_here
    fused = rrf_fuse(results_by_model={"siglip": r.points}, k=60, top_k=50)
    t_total = time.time() - t0
    times.append(t_total)
    print(f"  run #{i+1}: enc={t_enc_here*1000:.0f}ms qdrant={t_q*1000:.0f}ms rrf<1ms TOTAL={t_total*1000:.0f}ms, results={len(fused)}")
print(f"  AVG END-TO-END (single model): {sum(times)/len(times)*1000:.0f}ms")

print("\n" + "=" * 60)
print("BENCHMARK DONE")
