# HXQ Receipt Schema v2.0

## Overview

Every HXQ quantization operation produces a receipt: a JSON document proving
what the codec did, to what data, with what fidelity, at what cost. Receipts
are the contract between the codec and any downstream consumer.

## Schema Version

Every receipt MUST include `"schema_version": "2.0"`.

Receipts without a `schema_version` field are implicitly v1. Readers MUST
accept v1 receipts by translating: `model` -> `asset_id`,
`model_hf_id` -> `asset_source` (prepend `huggingface:`),
`name` (per-tensor) -> `tensor_id`.

## Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | YES | `"2.0"` |
| `experiment` | string | YES | Experiment identifier |
| `asset_id` | string | YES | Human-readable name of the asset |
| `asset_type` | string | YES | One of the defined asset types (see below) |
| `asset_source` | string | YES | Source URI in `<scheme>:<identifier>` format |
| `codec` | string | YES | Codec variant used (e.g. `"hxq_affine_6"`) |
| `bpw` | float | YES | Bits per weight/element |
| `method` | string | YES | Human-readable method description |
| `group_size` | int | YES | Elements per quantization group |
| `n_tensors` | int | YES | Number of tensors processed |
| `cos_min` | float | YES | Minimum cosine similarity across all tensors |
| `cos_mean` | float | YES | Mean cosine similarity |
| `n_pass` | int | YES | Tensors passing quality gate (cos >= threshold) |
| `n_fail` | int | YES | Tensors failing quality gate |
| `gate` | string | YES | `"PASS"` or `"FAIL"` |
| `gate_threshold` | float | YES | Cosine similarity threshold used |
| `per_tensor` | array | YES | Per-tensor results (see below) |
| `cost` | object | YES | Compute cost block (see below) |

## Codec Enum

| Value | bpw | Description |
|---|---|---|
| `hxq_affine_6` | 6.25 | 6-bit packed indices, fp16 scale/offset, group size 128 |
| `hxq_affine_g128` | 8.25 | uint8 indices, fp16 scale/offset, group size 128 |

Adding a new codec variant requires updating this spec.

## Asset Type Enum

| Value | Description |
|---|---|
| `llm_weight` | Weight tensor from a language model |
| `embedding_batch` | Batch of embedding vectors (e.g. sentence-transformers output) |
| `state_vector` | Runtime state (e.g. SSM hidden state, KV cache snapshot) |
| `merkle_leaves` | Leaf values from a Merkle tree or hash-based data structure |
| `generic_tensor` | Any tensor not covered by the above types |

Adding a new type requires updating this spec.

## Asset Source Format

Format: `<scheme>:<identifier>`

| Scheme | Example |
|---|---|
| `huggingface` | `huggingface:bert-base-uncased` |
| `local` | `local:/path/to/file.npz` |
| `dataset` | `dataset:msmarco-passages-1024-sample` |
| `url` | `url:https://example.com/data.bin` |
| `synthetic` | `synthetic:gaussian-768d-1024batch` |

## Per-Tensor Entry

| Field | Type | Required | Description |
|---|---|---|---|
| `tensor_id` | string | YES | Unique identifier for the tensor |
| `shape` | array[int] | YES | Tensor dimensions |
| `numel` | int | YES | Total number of elements |
| `cos_sim` | float | YES | Cosine similarity (original vs reconstructed) |
| `time_ms` | float | YES | Quantize+dequantize round-trip time in ms |

## Cost Block

Every receipt MUST include a cost block. A result without its cost is
an incomplete proof.

| Field | Type | Required | Description |
|---|---|---|---|
| `wall_time_s` | float | YES | Wall-clock time in seconds |
| `cpu_time_s` | float | YES | CPU time in seconds |
| `peak_memory_mb` | float | YES | Peak RSS in MB |
| `python_version` | string | YES | Python version (if applicable) |
| `hostname` | string | YES | Machine hostname |
| `timestamp_start` | string | YES | ISO 8601 start time |
| `timestamp_end` | string | YES | ISO 8601 end time |

## Operating Constraints

- HXQ operates on tensors of at least 128 elements organized into groups
  of 128.
- Tensors with fewer than 2 groups (256 elements) are not recommended;
  cosine similarity drops with group count below this threshold.
- The 6-bit variant requires `numel % 128 == 0` after any padding policy
  the caller defines.
- Padding policy is caller-defined. The codec zero-pads to group boundary
  by default. Quality gate results exclude padding elements.

## License

MIT
