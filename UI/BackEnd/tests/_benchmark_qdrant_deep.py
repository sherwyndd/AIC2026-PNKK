"""
Diagnostic script to profile Qdrant client vs HTTP latency in detail.
Measures:
1. TCP connect time
2. HTTP send time
3. Server processing time (from Qdrant response headers/payload)
4. Response receive & decode time
5. Comparison between with_payload=True vs with_payload=False
6. Comparison between python QdrantClient vs raw HTTP request vs curl
"""

import http.client
import json
import logging
import socket
import sys
import time
from pathlib import Path
import urllib.request

import numpy as np

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
for name in ("httpx", "httpcore", "qdrant_client", "urllib3"):
    logging.getLogger(name).setLevel(logging.DEBUG)

logger = logging.getLogger("qdrant_diagnostic")

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "image_siglip"

# 1. Generate random dummy embedding (dim 768)
np.random.seed(42)
dummy_vec = np.random.randn(768).astype(np.float32).tolist()


def test_raw_socket_timing():
    logger.info("=== 1. Low-level Socket & HTTP timing test ===")
    t_start = time.perf_counter()

    # Measure TCP connect
    t0 = time.perf_counter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120.0)
    sock.connect((QDRANT_HOST, QDRANT_PORT))
    t_tcp = time.perf_counter() - t0

    # Build HTTP POST payload
    body = json.dumps({
        "query": dummy_vec,
        "limit": 10,
        "with_payload": True,
        "params": {"hnsw_ef": 128}
    }).encode("utf-8")

    req = (
        f"POST /collections/{COLLECTION}/points/query HTTP/1.1\r\n"
        f"Host: {QDRANT_HOST}:{QDRANT_PORT}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8") + body

    # Measure send
    t1 = time.perf_counter()
    sock.sendall(req)
    t_send = time.perf_counter() - t1

    # Measure time to first byte (TTFB) / server processing
    t2 = time.perf_counter()
    res_raw = b""
    first_byte = sock.recv(1)
    t_ttfb = time.perf_counter() - t2
    res_raw += first_byte

    # Measure receive rest of body
    t3 = time.perf_counter()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        res_raw += chunk
    t_recv = time.perf_counter() - t3
    sock.close()

    # Measure decode
    t4 = time.perf_counter()
    header_part, _, body_part = res_raw.partition(b"\r\n\r\n")
    data = json.loads(body_part.decode("utf-8"))
    t_decode = time.perf_counter() - t4
    t_total = time.perf_counter() - t_start

    server_time = data.get("time", 0.0)
    points_cnt = len(data.get("result", {}).get("points", []))

    print("\n--- RAW SOCKET BREAKDOWN (with_payload=True, limit=10) ---")
    print(f"  TCP Connect:       {t_tcp*1000:.3f} ms")
    print(f"  Send Request:      {t_send*1000:.3f} ms")
    print(f"  Server TTFB:       {t_ttfb*1000:.3f} ms (Server reported compute: {server_time*1000:.3f} ms)")
    print(f"  Receive Body:      {t_recv*1000:.3f} ms (Points: {points_cnt})")
    print(f"  JSON Decode:       {t_decode*1000:.3f} ms")
    print(f"  TOTAL Wall Time:   {t_total*1000:.3f} ms\n")


def test_payload_vs_no_payload():
    logger.info("=== 2. Comparing with_payload=True vs with_payload=False ===")
    for with_pl in (False, True):
        for limit in (10, 100, 300):
            t0 = time.perf_counter()
            body = json.dumps({
                "query": dummy_vec,
                "limit": limit,
                "with_payload": with_pl,
                "params": {"hnsw_ef": 128}
            }).encode("utf-8")
            req = urllib.request.Request(
                f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}/points/query",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            server_time = res.get("time", 0.0)
            pts = len(res.get("result", {}).get("points", []))
            print(f"  Limit={limit:3d} | with_payload={str(with_pl):5s} | Wall Time: {elapsed*1000:8.2f} ms | Qdrant Internal Time: {server_time*1000:8.2f} ms | Points: {pts}")


def test_qdrant_client_sdk():
    logger.info("=== 3. Testing official QdrantClient Python SDK ===")
    from qdrant_client import QdrantClient
    from qdrant_client.models import SearchParams

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    for limit in (10, 60, 120):
        t0 = time.perf_counter()
        res = client.query_points(
            collection_name=COLLECTION,
            query=dummy_vec,
            limit=limit,
            search_params=SearchParams(hnsw_ef=128),
            timeout=120
        )
        elapsed = time.perf_counter() - t0
        pts = len(res.points) if hasattr(res, "points") else len(res)
        print(f"  QdrantClient.query_points(limit={limit:3d}) | Wall Time: {elapsed*1000:8.2f} ms | Returned Points: {pts}")


if __name__ == "__main__":
    test_raw_socket_timing()
    test_payload_vs_no_payload()
    test_qdrant_client_sdk()
