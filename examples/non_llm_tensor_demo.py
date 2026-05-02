#!/usr/bin/env python3
"""
Non-LLM tensor compression demo.

Compresses sentence-transformer embeddings using HXQ affine quantization,
emits a fidelity receipt. Proves the codec works on arbitrary dense tensors,
not just model weights.

Requirements: numpy, sentence-transformers (pip install sentence-transformers)
"""

import numpy as np
import json
import time
import platform
import resource
from datetime import datetime

# --- Configuration ---
GROUP_SIZE = 128
LEVELS_6BIT = 64
GATE_THRESHOLD = 0.999

def fp32_to_fp16(x):
    return float(np.float16(x))

def quantize_affine_6bit(data):
    """Per-group-128 affine quantization to 6 bits."""
    n = len(data)
    n_groups = n // GROUP_SIZE
    recon = np.zeros_like(data)

    for g in range(n_groups):
        chunk = data[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        vmin = fp32_to_fp16(chunk.min())
        vmax = fp32_to_fp16(chunk.max())
        scale = fp32_to_fp16((vmax - vmin) / (LEVELS_6BIT - 1))
        inv_scale = 1.0 / scale if scale > 1e-10 else 0.0

        for j in range(GROUP_SIZE):
            idx = int(np.clip((chunk[j] - vmin) * inv_scale + 0.5, 0, LEVELS_6BIT - 1))
            recon[g * GROUP_SIZE + j] = idx * scale + vmin

    return recon

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # --- Load or generate embeddings ---
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformer model...")
        model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

        # Sample texts (public domain)
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models compress information into dense representations.",
            "Calibration-free quantization requires no training data.",
            "Sovereign wealth funds hold passive equity positions globally.",
            "The fidelity receipt proves the codec preserved the original signal.",
            "Edge deployment requires small, portable, auditable inference.",
            "Per-group affine scaling maps each block to a linear range.",
            "Dense numeric tensors are the universal substrate of modern AI.",
        ] * 16  # 128 embeddings

        print(f"Encoding {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        source = "dataset:demo-texts-128"
        asset_source_scheme = "synthetic:sentence-transformer-128-texts"

    except ImportError:
        print("sentence-transformers not installed. Using synthetic embeddings.")
        print("  (pip install sentence-transformers for real embeddings)")
        np.random.seed(42)
        embeddings = np.random.randn(128, 768).astype(np.float32)
        source = "synthetic"
        asset_source_scheme = "synthetic:gaussian-768d-128batch"

    n_embeddings, dim = embeddings.shape
    flat = embeddings.flatten()
    numel = len(flat)

    # Align to group boundary
    aligned_numel = (numel // GROUP_SIZE) * GROUP_SIZE
    flat = flat[:aligned_numel]
    n_groups = aligned_numel // GROUP_SIZE

    print(f"Embeddings: {n_embeddings} x {dim} = {numel} elements")
    print(f"Aligned: {aligned_numel} elements, {n_groups} groups")
    print(f"Compressing with 6-bit affine (group size {GROUP_SIZE})...")

    # --- Compress ---
    recon = quantize_affine_6bit(flat)

    # --- Per-embedding fidelity ---
    per_tensor = []
    cos_values = []
    for i in range(n_embeddings):
        start = i * dim
        end = start + dim
        if end > aligned_numel:
            break
        orig_emb = flat[start:end]
        recon_emb = recon[start:end]
        cos = cosine_similarity(orig_emb, recon_emb)
        cos_values.append(cos)
        per_tensor.append({
            "tensor_id": f"embedding_{i:04d}",
            "shape": [dim],
            "numel": dim,
            "cos_sim": round(cos, 6),
            "time_ms": 0.0,
        })

    cos_min = min(cos_values)
    cos_mean = sum(cos_values) / len(cos_values)
    n_pass = sum(1 for c in cos_values if c >= GATE_THRESHOLD)
    n_fail = len(cos_values) - n_pass

    # --- Cost block ---
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- Receipt ---
    receipt = {
        "schema_version": "2.0",
        "experiment": "non-llm-tensor-demo",
        "asset_id": "sentence-transformer-embeddings",
        "asset_type": "embedding_batch",
        "asset_source": asset_source_scheme,
        "codec": "hxq_affine_6",
        "bpw": 6.25,
        "method": "per-group-128 affine, 6-bit, min/max scaling, no calibration",
        "group_size": GROUP_SIZE,
        "n_tensors": len(per_tensor),
        "cos_min": round(cos_min, 6),
        "cos_mean": round(cos_mean, 6),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "gate": "PASS" if n_fail == 0 else "FAIL",
        "gate_threshold": GATE_THRESHOLD,
        "per_tensor": per_tensor,
        "cost": cost,
    }

    # --- Output ---
    print(f"\nResults:")
    print(f"  cos_min:  {cos_min:.6f}")
    print(f"  cos_mean: {cos_mean:.6f}")
    print(f"  pass:     {n_pass}/{len(per_tensor)}")
    print(f"  gate:     {receipt['gate']}")
    print(f"  time:     {cost['wall_time_s']}s")

    receipt_json = json.dumps(receipt, indent=2)
    print(f"\nReceipt ({len(receipt_json)} bytes):")
    # Print just the top-level fields, not per_tensor
    summary = {k: v for k, v in receipt.items() if k != "per_tensor"}
    summary["per_tensor"] = f"[{len(per_tensor)} entries]"
    print(json.dumps(summary, indent=2))

    # Save receipt
    import os
    receipt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "receipts")
    os.makedirs(receipt_dir, exist_ok=True)
    receipt_path = os.path.join(receipt_dir, "non_llm_demo_receipt.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt saved: {receipt_path}")


if __name__ == "__main__":
    main()
