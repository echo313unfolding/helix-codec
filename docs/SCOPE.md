# HXQ Scope and Limits

## What HXQ is

HXQ is a calibration-free tensor codec for dense numeric tensors. It compresses
arrays of floats into 6-bit or 8-bit representations using per-group affine
quantization (min/max scaling within groups of 128 elements).

It does not require calibration data, training data, or importance matrices.
Every quantization operation emits a receipt proving fidelity.

## What HXQ works on

Any dense numeric tensor with sufficient elements:

- LLM weight tensors
- Encoder/vision model weights
- Embedding batches (sentence-transformers, retrieval systems)
- State vectors (SSM hidden states, KV cache snapshots)
- Risk/feature vectors
- Any array of floats where group-level affine scaling is a reasonable model

## What HXQ does not do

- **Not a universal compression algorithm.** HXQ compresses tensors, not
  arbitrary byte streams, images, audio, or text.
- **Not a cryptographic proof system.** Receipts prove fidelity metrics
  (cosine similarity, element count, timing), not tamper-resistance or
  zero-knowledge properties.
- **Not a training-aware quantizer.** There is no calibration pass, no
  importance weighting, no activation-aware scaling. This is the tradeoff:
  zero data dependency in exchange for slightly higher quantization error
  compared to calibration-based methods.
- **Not optimal for sparse or degenerate tensors.** Tensors with many
  exact zeros, extreme outliers, or fewer than 256 elements may produce
  poor fidelity.

## Group size constraint

HXQ operates on groups of 128 elements. This means:

- Input must have at least 128 elements (1 group).
- Tensors with fewer than 256 elements (2 groups) are not recommended;
  cosine similarity degrades below this threshold.
- The 6-bit variant requires `numel % 128 == 0`.
- For tensors that are not group-aligned, the caller must choose a policy:
  - **Pad:** zero-pad to the next group boundary (default in C API).
  - **Skip:** store the tensor uncompressed.
  - **Exact-store:** bypass the codec entirely.
  - **Smaller group size:** not currently supported; would require a new
    codec variant.

## Bits per weight

| Variant | bpw | Block layout |
|---|---|---|
| `hxq_affine_6` | 6.25 | 96 packed bytes + 4 bytes (fp16 scale + fp16 offset) = 100 bytes / 128 elements |
| `hxq_affine_g128` | 8.25 | 128 uint8 indices + 4 bytes (fp16 scale + fp16 offset) = 132 bytes / 128 elements |

Both variants use fp16 for scale and offset. Quantization error comes from
two sources: fp16 rounding of the affine parameters and index quantization
(64 levels for 6-bit, 256 levels for 8-bit).

## When to use something else

- If you have calibration data and want minimum perplexity loss: use GPTQ,
  AWQ, or similar calibration-based quantizers.
- If your tensors are sparse: use sparse storage formats.
- If your tensors have fewer than 128 elements: store them exact.
- If you need cryptographic integrity: layer a hash or signature over the
  receipt and compressed bytes.
