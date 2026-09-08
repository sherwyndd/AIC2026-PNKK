"""Test script to measure latency of SigLIP2 encoding step by step."""

import time
import torch
from models.siglip_encoder import load_siglip, encode_text_siglip

print("=== Loading SigLIP2 model ===")
start_time = time.time()
load_siglip(device="cuda:0")
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f}s")

print("\n=== Testing encoding latency ===")
test_queries = [
    "cô gái mặc áo đỏ",
    "người đi xe máy",
    "a",
    "test query with some more words to check if length matters"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    
    # Full encoding time
    start = time.time()
    emb = encode_text_siglip(query)
    full_time = time.time() - start
    print(f"  Full encoding: {full_time:.4f}s")
    print(f"  Embedding shape: {emb.shape}")

print("\n=== Test completed ===")
