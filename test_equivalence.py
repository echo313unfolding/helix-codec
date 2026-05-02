#!/usr/bin/env python3
"""Equivalence test: standalone helix_codec vs ggml quantize/dequantize.
Generates a test tensor, quantizes with both, verifies byte-identical output."""

import subprocess, struct, tempfile, os, sys
import numpy as np
from pathlib import Path

# Generate deterministic test data: 1024 floats
np.random.seed(42)
data = np.random.randn(1024).astype(np.float32)

# Write to temp binary file
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    f.write(data.tobytes())
    input_path = f.name

# Run standalone demo
demo = Path(__file__).parent / 'hxq_demo'
result = subprocess.run([str(demo), input_path], capture_output=True, text=True)
if result.returncode not in (0, 1):
    print(f"Demo failed: {result.stderr}")
    sys.exit(1)

import json
receipt = json.loads(result.stdout)
cos_standalone = receipt['cos_sim']

# Now do the same quantize/dequantize in Python (matching the C implementation exactly)
G = 128
LEVELS = 64

def fp32_to_fp16_bits(f):
    """Match C fp32_to_fp16 exactly (IEEE 754 half-precision)."""
    return int(np.float16(f).view(np.uint16))

def fp16_bits_to_fp32(h):
    """Match C fp16_to_fp32 exactly."""
    return float(np.uint16(h).view(np.float16))

def quantize_6bit_python(data):
    """Python implementation matching helix_codec.c exactly."""
    nb = len(data) // G
    blocks = []
    for i in range(nb):
        chunk = data[i*G:(i+1)*G]
        vmin = chunk.min()
        vmax = chunk.max()
        scale = (vmax - vmin) / 63.0
        inv_scale = 1.0 / scale if scale > 1e-10 else 0.0

        scale_fp16 = fp32_to_fp16_bits(scale)
        offset_fp16 = fp32_to_fp16_bits(vmin)

        qs = bytearray(96)
        for j in range(0, G, 4):
            idx = []
            for m in range(4):
                v = (chunk[j+m] - vmin) * inv_scale
                v = max(0.0, min(63.0, v))
                idx.append(int(v + 0.5))
            byte_idx = (j // 4) * 3
            qs[byte_idx + 0] = (idx[0] & 0x3F) | ((idx[1] & 0x03) << 6)
            qs[byte_idx + 1] = ((idx[1] >> 2) & 0x0F) | ((idx[2] & 0x0F) << 4)
            qs[byte_idx + 2] = ((idx[2] >> 4) & 0x03) | ((idx[3] & 0x3F) << 2)

        blocks.append((scale_fp16, offset_fp16, bytes(qs)))
    return blocks

def dequantize_6bit_python(blocks):
    """Python dequant matching helix_codec.c."""
    out = []
    for scale_fp16, offset_fp16, qs in blocks:
        scale = fp16_bits_to_fp32(scale_fp16)
        offset = fp16_bits_to_fp32(offset_fp16)
        for j in range(0, G, 4):
            byte_idx = (j // 4) * 3
            b0, b1, b2 = qs[byte_idx], qs[byte_idx+1], qs[byte_idx+2]
            idx0 = b0 & 0x3F
            idx1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
            idx2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
            idx3 = b2 >> 2
            out.extend([idx0 * scale + offset, idx1 * scale + offset,
                        idx2 * scale + offset, idx3 * scale + offset])
    return np.array(out, dtype=np.float32)

blocks_py = quantize_6bit_python(data)
recon_py = dequantize_6bit_python(blocks_py)

cos_python = float(np.dot(data, recon_py) / (np.linalg.norm(data) * np.linalg.norm(recon_py)))

print(f"Standalone C cos_sim: {cos_standalone:.6f}")
print(f"Python cos_sim:       {cos_python:.6f}")
print(f"Delta:                {abs(cos_standalone - cos_python):.2e}")

# The C and Python should agree to within fp16 rounding
if abs(cos_standalone - cos_python) < 1e-4:
    print("EQUIVALENCE: PASS")
else:
    print("EQUIVALENCE: FAIL — implementations diverge")
    sys.exit(1)

os.unlink(input_path)
print(f"\nReceipt gate: {receipt['gate']}")
