/*
 * helix_codec.h — HXQ calibration-free tensor codec
 *
 * Compresses arrays of floats into 6-bit or 8-bit representations using
 * per-group affine quantization (min/max scaling). No calibration data,
 * no training data, no importance matrices. Produces a per-tensor receipt
 * proving fidelity to the original.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Echo Labs
 */

#ifndef HELIX_CODEC_H
#define HELIX_CODEC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- constants ---------- */

#define HXQ_GROUP_SIZE  128
#define HXQ_6BIT_PACKED 96   /* 128 * 6 / 8 */

/* ---------- status codes ---------- */

typedef enum {
    HXQ_OK                       = 0,
    HXQ_ERR_NULL_POINTER         = 1,
    HXQ_ERR_NUMEL_TOO_SMALL      = 2,
    HXQ_ERR_NUMEL_NOT_ALIGNED    = 3,
} hxq_status;

/* ---------- block structs ---------- */

/* 8-bit affine: fp16 scale + fp16 offset + 128 × uint8 = 132 bytes, 8.25 bpw */
typedef struct {
    uint16_t scale;    /* fp16 */
    uint16_t offset;   /* fp16 */
    uint8_t  qs[HXQ_GROUP_SIZE];
} hxq_block_g128;

/* 6-bit affine: fp16 scale + fp16 offset + 96 packed bytes = 100 bytes, 6.25 bpw */
typedef struct {
    uint16_t scale;    /* fp16 */
    uint16_t offset;   /* fp16 */
    uint8_t  qs[HXQ_6BIT_PACKED];
} hxq_block_6;

/* ---------- receipt ---------- */

typedef struct {
    /* Set by caller before quantize */
    const char * asset_id;
    const char * asset_type;     /* "llm_weight", "embedding_batch", etc. */
    const char * asset_source;   /* "huggingface:...", "dataset:...", etc. */
    const char * tensor_id;

    /* Filled by quantize */
    size_t  numel;
    size_t  n_groups;
    float   cos_sim;
    float   bpw;
    int     pass;        /* 1 if cos_sim >= gate_threshold */
    float   gate_threshold;
    double  time_ms;
} hxq_receipt;

/* ---------- API ---------- */

/*
 * Quantize numel floats into 6-bit affine blocks.
 * numel must be >= HXQ_GROUP_SIZE and a multiple of HXQ_GROUP_SIZE.
 * out must point to numel/HXQ_GROUP_SIZE blocks.
 * receipt_out may be NULL; if non-NULL, filled with quality metrics.
 */
hxq_status hxq_quantize_6bit(const float * in, size_t numel,
                              hxq_block_6 * out, hxq_receipt * receipt_out);

hxq_status hxq_dequantize_6bit(const hxq_block_6 * in, size_t n_blocks,
                                float * out);

/*
 * Quantize numel floats into 8-bit (g128) affine blocks.
 * Same alignment requirements as 6-bit.
 */
hxq_status hxq_quantize_g128(const float * in, size_t numel,
                              hxq_block_g128 * out, hxq_receipt * receipt_out);

hxq_status hxq_dequantize_g128(const hxq_block_g128 * in, size_t n_blocks,
                                float * out);

/*
 * Serialize receipt to JSON. Returns number of bytes written (excluding
 * null terminator), or required size if buf is NULL or buflen is too small.
 */
size_t hxq_receipt_to_json(const hxq_receipt * r, char * buf, size_t buflen);

/*
 * Human-readable status string.
 */
const char * hxq_status_str(hxq_status s);

#ifdef __cplusplus
}
#endif

#endif /* HELIX_CODEC_H */
