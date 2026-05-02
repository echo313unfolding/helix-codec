/*
 * helix_codec.c — HXQ calibration-free tensor codec
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Echo Labs
 */

#include "helix_codec.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ---------- fp16 helpers (from ggml, public domain) ---------- */

static inline float fp32_from_bits(uint32_t w) {
    union { uint32_t as_bits; float as_value; } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union { float as_value; uint32_t as_bits; } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline uint16_t fp32_to_fp16(float f) {
    const float scale_to_inf  = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

/* ---------- cosine similarity ---------- */

static float cosine_similarity(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    na = sqrt(na);
    nb = sqrt(nb);
    if (na < 1e-10 || nb < 1e-10) return 0.0f;
    return (float)(dot / (na * nb));
}

/* ---------- validation ---------- */

static hxq_status validate_input(const float * in, size_t numel, const void * out) {
    if (!in || !out) return HXQ_ERR_NULL_POINTER;
    if (numel < HXQ_GROUP_SIZE) return HXQ_ERR_NUMEL_TOO_SMALL;
    if (numel % HXQ_GROUP_SIZE != 0) return HXQ_ERR_NUMEL_NOT_ALIGNED;
    return HXQ_OK;
}

/* ---------- 8-bit (g128) ---------- */

hxq_status hxq_quantize_g128(const float * in, size_t numel,
                              hxq_block_g128 * out, hxq_receipt * receipt_out) {
    hxq_status s = validate_input(in, numel, out);
    if (s != HXQ_OK) return s;

    const size_t nb = numel / HXQ_GROUP_SIZE;

    for (size_t i = 0; i < nb; i++) {
        float vmin =  FLT_MAX;
        float vmax = -FLT_MAX;

        for (int j = 0; j < HXQ_GROUP_SIZE; j++) {
            const float v = in[i * HXQ_GROUP_SIZE + j];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }

        const float range = vmax - vmin;
        const float scale = range / 255.0f;
        const float inv_scale = (scale > 1e-10f) ? 1.0f / scale : 0.0f;

        out[i].scale  = fp32_to_fp16(scale);
        out[i].offset = fp32_to_fp16(vmin);

        for (int j = 0; j < HXQ_GROUP_SIZE; j++) {
            float v = (in[i * HXQ_GROUP_SIZE + j] - vmin) * inv_scale;
            v = (v < 0.0f) ? 0.0f : (v > 255.0f) ? 255.0f : v;
            out[i].qs[j] = (uint8_t)(v + 0.5f);
        }
    }

    if (receipt_out) {
        /* Dequantize into temp buffer and compute cos_sim */
        float * tmp = (float *)malloc(numel * sizeof(float));
        if (tmp) {
            hxq_dequantize_g128(out, nb, tmp);
            receipt_out->numel = numel;
            receipt_out->n_groups = nb;
            receipt_out->cos_sim = cosine_similarity(in, tmp, numel);
            receipt_out->bpw = 8.25f;
            receipt_out->gate_threshold = 0.999f;
            receipt_out->pass = (receipt_out->cos_sim >= receipt_out->gate_threshold) ? 1 : 0;
            free(tmp);
        }
    }

    return HXQ_OK;
}

hxq_status hxq_dequantize_g128(const hxq_block_g128 * in, size_t n_blocks,
                                float * out) {
    if (!in || !out) return HXQ_ERR_NULL_POINTER;

    for (size_t i = 0; i < n_blocks; i++) {
        const float scale  = fp16_to_fp32(in[i].scale);
        const float offset = fp16_to_fp32(in[i].offset);

        for (int j = 0; j < HXQ_GROUP_SIZE; j++) {
            out[i * HXQ_GROUP_SIZE + j] = in[i].qs[j] * scale + offset;
        }
    }

    return HXQ_OK;
}

/* ---------- 6-bit ---------- */

hxq_status hxq_quantize_6bit(const float * in, size_t numel,
                              hxq_block_6 * out, hxq_receipt * receipt_out) {
    hxq_status s = validate_input(in, numel, out);
    if (s != HXQ_OK) return s;

    const size_t nb = numel / HXQ_GROUP_SIZE;

    for (size_t i = 0; i < nb; i++) {
        float vmin =  FLT_MAX;
        float vmax = -FLT_MAX;

        for (int j = 0; j < HXQ_GROUP_SIZE; j++) {
            const float v = in[i * HXQ_GROUP_SIZE + j];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }

        const float range = vmax - vmin;
        const float scale = range / 63.0f;
        const float inv_scale = (scale > 1e-10f) ? 1.0f / scale : 0.0f;

        out[i].scale  = fp32_to_fp16(scale);
        out[i].offset = fp32_to_fp16(vmin);

        /* Pack 4 indices per 3 bytes: [aaaaaabb|bbbbcccc|ccdddddd] */
        for (int j = 0; j < HXQ_GROUP_SIZE; j += 4) {
            uint8_t idx[4];
            for (int m = 0; m < 4; m++) {
                float v = (in[i * HXQ_GROUP_SIZE + j + m] - vmin) * inv_scale;
                v = (v < 0.0f) ? 0.0f : (v > 63.0f) ? 63.0f : v;
                idx[m] = (uint8_t)(v + 0.5f);
            }
            const int byte_idx = (j / 4) * 3;
            out[i].qs[byte_idx + 0] = (idx[0] & 0x3F) | ((idx[1] & 0x03) << 6);
            out[i].qs[byte_idx + 1] = ((idx[1] >> 2) & 0x0F) | ((idx[2] & 0x0F) << 4);
            out[i].qs[byte_idx + 2] = ((idx[2] >> 4) & 0x03) | ((idx[3] & 0x3F) << 2);
        }
    }

    if (receipt_out) {
        float * tmp = (float *)malloc(numel * sizeof(float));
        if (tmp) {
            hxq_dequantize_6bit(out, nb, tmp);
            receipt_out->numel = numel;
            receipt_out->n_groups = nb;
            receipt_out->cos_sim = cosine_similarity(in, tmp, numel);
            receipt_out->bpw = 6.25f;
            receipt_out->gate_threshold = 0.999f;
            receipt_out->pass = (receipt_out->cos_sim >= receipt_out->gate_threshold) ? 1 : 0;
            free(tmp);
        }
    }

    return HXQ_OK;
}

hxq_status hxq_dequantize_6bit(const hxq_block_6 * in, size_t n_blocks,
                                float * out) {
    if (!in || !out) return HXQ_ERR_NULL_POINTER;

    for (size_t i = 0; i < n_blocks; i++) {
        const float scale  = fp16_to_fp32(in[i].scale);
        const float offset = fp16_to_fp32(in[i].offset);

        for (int j = 0; j < HXQ_GROUP_SIZE; j += 4) {
            const int byte_idx = (j / 4) * 3;
            const uint8_t b0 = in[i].qs[byte_idx + 0];
            const uint8_t b1 = in[i].qs[byte_idx + 1];
            const uint8_t b2 = in[i].qs[byte_idx + 2];

            const uint8_t idx0 =  b0       & 0x3F;
            const uint8_t idx1 = ((b0 >> 6) | (b1 << 2)) & 0x3F;
            const uint8_t idx2 = ((b1 >> 4) | (b2 << 4)) & 0x3F;
            const uint8_t idx3 =  b2 >> 2;

            out[i * HXQ_GROUP_SIZE + j + 0] = idx0 * scale + offset;
            out[i * HXQ_GROUP_SIZE + j + 1] = idx1 * scale + offset;
            out[i * HXQ_GROUP_SIZE + j + 2] = idx2 * scale + offset;
            out[i * HXQ_GROUP_SIZE + j + 3] = idx3 * scale + offset;
        }
    }

    return HXQ_OK;
}

/* ---------- receipt JSON ---------- */

size_t hxq_receipt_to_json(const hxq_receipt * r, char * buf, size_t buflen) {
    if (!r) return 0;

    char tmp[2048];
    int n = snprintf(tmp, sizeof(tmp),
        "{\n"
        "  \"schema_version\": \"2.0\",\n"
        "  \"asset_id\": \"%s\",\n"
        "  \"asset_type\": \"%s\",\n"
        "  \"asset_source\": \"%s\",\n"
        "  \"codec\": \"%s\",\n"
        "  \"bpw\": %.2f,\n"
        "  \"group_size\": %d,\n"
        "  \"tensor_id\": \"%s\",\n"
        "  \"numel\": %zu,\n"
        "  \"n_groups\": %zu,\n"
        "  \"cos_sim\": %.6f,\n"
        "  \"gate_threshold\": %.3f,\n"
        "  \"gate\": \"%s\",\n"
        "  \"time_ms\": %.3f\n"
        "}\n",
        r->asset_id    ? r->asset_id    : "",
        r->asset_type  ? r->asset_type  : "",
        r->asset_source? r->asset_source: "",
        r->bpw > 7.0f ? "hxq_affine_g128" : "hxq_affine_6",
        (double)r->bpw,
        HXQ_GROUP_SIZE,
        r->tensor_id   ? r->tensor_id   : "",
        r->numel,
        r->n_groups,
        (double)r->cos_sim,
        (double)r->gate_threshold,
        r->pass ? "PASS" : "FAIL",
        r->time_ms
    );

    if (n < 0) return 0;
    size_t needed = (size_t)n + 1;
    if (buf && buflen >= needed) {
        memcpy(buf, tmp, needed);
    }
    return (size_t)n;
}

/* ---------- status string ---------- */

const char * hxq_status_str(hxq_status s) {
    switch (s) {
        case HXQ_OK:                    return "ok";
        case HXQ_ERR_NULL_POINTER:      return "null pointer";
        case HXQ_ERR_NUMEL_TOO_SMALL:   return "numel < 128 (minimum one group)";
        case HXQ_ERR_NUMEL_NOT_ALIGNED: return "numel not a multiple of 128";
        default:                        return "unknown error";
    }
}
