# GGUF/GGML as a Downstream Consumer

## Relationship

HXQ is a standalone tensor codec. GGUF/GGML is one downstream runtime that
consumes HXQ-compressed tensors. The codec does not depend on GGUF, llama.cpp,
or any inference runtime.

```
helix-codec (standalone C library)
    |
    v
GGUF file format (stores HXQ blocks as native quantization types)
    |
    v
llama.cpp / ggml (decodes HXQ blocks during inference)
```

## How it works

The [llama.cpp fork](https://github.com/echo313unfolding/llama.cpp) on branch
`hxq-affine-type` registers two native quantization types:

| GGML type | Codec variant | Block size | bpw |
|---|---|---|---|
| `GGML_TYPE_HXQ_AFFINE_6` | `hxq_affine_6` | 100 bytes | 6.25 |
| `GGML_TYPE_HXQ_AFFINE_G128` | `hxq_affine_g128` | 132 bytes | 8.25 |

These types are wired into the GGML type table, the GGUF serializer, and the
GPU dequantization kernels (mmvq path). The dequantization math in llama.cpp
is identical to `hxq_dequantize_6bit` / `hxq_dequantize_g128` in this repo.

## Performance

Measured on Qwen2.5-Coder-3B, RTX 3090:

| Type | GPU decode tok/s | VRAM |
|---|---|---|
| HXQ_AFFINE_6 | 27.83 | 1.92 GB |
| HXQ_AFFINE_G128 | 27.83 | 2.15 GB |
| Q8_0 (baseline) | 28.32 | 3.57 GB |
| Q4_K_M (baseline) | 44.0 | 2.03 GB |

GPU decode reaches Q8_0 parity at 26% smaller size. CPU decode is slower
due to 6-bit SIMD unpack cost (see README for CPU numbers).

## What this proves

GGUF integration demonstrates that HXQ blocks survive a real inference
runtime: the compressed representation can be loaded, dequantized on the
fly during matrix multiplication, and produce correct model output. This is
a stronger claim than "the codec round-trips correctly" because it tests
the blocks under actual computational load with accumulated numerical error
across hundreds of layers.

## Building GGUF models with HXQ

The converter lives in the llama.cpp fork, not in this repo:

```bash
# In the llama.cpp fork (hxq-affine-type branch):
python3 convert_hf_to_gguf.py \
    --model-dir /path/to/hf-model \
    --outtype hxq_affine_6 \
    --outfile model-hxq6.gguf
```

This repo provides the codec math. The GGUF tooling is downstream.
