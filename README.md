# helix-codec

HXQ is a calibration-free tensor codec. It compresses arrays of floats — model
weights, embedding batches, state vectors, any dense tensor — into 6-bit or
8-bit representations with per-group affine quantization. It does not require
calibration data, does not access training data during quantization, and
produces a per-tensor receipt proving fidelity to the original.

## Non-LLM proof

The codec applied to 1024 sentence-transformer embeddings (MS MARCO passages,
`all-mpnet-base-v2`, 768-dim). 786,432 elements, 1024/1024 pass, cos_min
0.999558. The codec did not know it was compressing embeddings.

Receipt: [`receipts/hxq_non_llm_embedding_proof.json`](receipts/hxq_non_llm_embedding_proof.json)

## Downstream consumers

- **[llama.cpp fork](https://github.com/echo313unfolding/llama.cpp)** — GGML_TYPE_HXQ_AFFINE_6
  and GGML_TYPE_HXQ_AFFINE_G128 wired as native quantization types. LLM weight
  compression, 27.83 tok/s GPU decode (Q8_0 parity).
- **[sentinel-hybrid-stack](https://github.com/echo313unfolding/sentinel-hybrid-stack)** —
  3-tier security triage agent deployed on edge hardware with HXQ-compressed models.
- **Non-LLM tensor receipt** — 1024 MS MARCO sentence-transformer embeddings
  compressed and verified ([receipt](receipts/hxq_non_llm_embedding_proof.json)).
  Proves the codec works on arbitrary tensors, not just model weights.

## Build

```
make
```

Produces `libhelix_codec.a` and `hxq_demo`. No dependencies beyond a C99
compiler and `-lm`.

## Usage

```c
#include "helix_codec.h"

hxq_block_6 blocks[numel / HXQ_GROUP_SIZE];
hxq_receipt receipt = {
    .asset_id     = "my-embeddings",
    .asset_type   = "embedding_batch",
    .asset_source = "dataset:my-corpus-v1",
    .tensor_id    = "batch_0",
};

hxq_status s = hxq_quantize_6bit(data, numel, blocks, &receipt);
// receipt.cos_sim, receipt.pass, receipt.gate_threshold now filled
// The library fills hxq_receipt. Consumers may serialize the struct
// directly or use hxq_receipt_to_json for a default v2-schema rendering.
```

## API

```c
hxq_status hxq_quantize_6bit(const float* in, size_t numel,
                              hxq_block_6* out, hxq_receipt* receipt_out);
hxq_status hxq_dequantize_6bit(const hxq_block_6* in, size_t n_blocks,
                                float* out);
hxq_status hxq_quantize_g128(const float* in, size_t numel,
                              hxq_block_g128* out, hxq_receipt* receipt_out);
hxq_status hxq_dequantize_g128(const hxq_block_g128* in, size_t n_blocks,
                                float* out);
size_t     hxq_receipt_to_json(const hxq_receipt* r, char* buf, size_t buflen);
const char* hxq_status_str(hxq_status s);
```

## Variants

| Variant | bpw | Block size | Description |
|---|---|---|---|
| `hxq_affine_6` | 6.25 | 100 bytes / 128 elements | 6-bit packed indices + fp16 scale/offset |
| `hxq_affine_g128` | 8.25 | 132 bytes / 128 elements | uint8 indices + fp16 scale/offset |

## Constraints

- HXQ operates on tensors of at least 128 elements organized into groups
  of 128.
- Tensors with fewer than 2 groups (256 elements) are not recommended;
  cosine similarity drops with group count below this threshold.
- The 6-bit variant requires `numel % 128 == 0` after any padding policy
  the caller defines.

## Performance

Reference performance on Intel i9-9880H, 8 threads, Qwen2.5-3B:

| Variant | CPU tok/s | GPU tok/s (T2000) |
|---|---|---|
| `hxq_affine_6` | 3.09 | 27.83 |
| `hxq_affine_g128` | 8.94 | 27.83 |
| Q4_K_M (baseline) | 15.25 | 44.0 |

CPU performance is limited by 6-bit SIMD unpack cost. GPU decode (via
llama.cpp mmvq kernel) reaches Q8_0 parity. For CPU-only deployments,
`hxq_affine_g128` (8-bit) is recommended.

## Receipt schema

See [SCHEMA.md](SCHEMA.md) for the v2.0 receipt schema specification.

## License

MIT
