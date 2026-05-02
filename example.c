/*
 * example.c — HXQ codec demo
 * Usage: ./hxq_demo input.bin > receipt.json
 *        ./hxq_demo            (uses built-in test data)
 */

#include "helix_codec.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char ** argv) {
    float * data = NULL;
    size_t numel = 0;

    if (argc > 1) {
        FILE * f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        fseek(f, 0, SEEK_END);
        numel = (size_t)ftell(f) / sizeof(float);
        numel = (numel / HXQ_GROUP_SIZE) * HXQ_GROUP_SIZE; /* align */
        fseek(f, 0, SEEK_SET);
        data = (float *)malloc(numel * sizeof(float));
        if (fread(data, sizeof(float), numel, f) != numel) {
            fprintf(stderr, "short read\n"); free(data); fclose(f); return 1;
        }
        fclose(f);
    } else {
        /* Built-in test: 1024 floats (8 groups) */
        numel = 1024;
        data = (float *)malloc(numel * sizeof(float));
        for (size_t i = 0; i < numel; i++)
            data[i] = sinf((float)i * 0.01f) * 2.0f;
    }

    size_t n_blocks = numel / HXQ_GROUP_SIZE;
    hxq_block_6 * blocks = (hxq_block_6 *)malloc(n_blocks * sizeof(hxq_block_6));
    hxq_receipt receipt = {
        .asset_id     = argc > 1 ? argv[1] : "built-in-test",
        .asset_type   = "generic_tensor",
        .asset_source = argc > 1 ? "local:input.bin" : "synthetic:sin-1024",
        .tensor_id    = "data",
    };

    hxq_status s = hxq_quantize_6bit(data, numel, blocks, &receipt);
    if (s != HXQ_OK) {
        fprintf(stderr, "quantize failed: %s\n", hxq_status_str(s));
        free(data); free(blocks);
        return 1;
    }

    char json[2048];
    hxq_receipt_to_json(&receipt, json, sizeof(json));
    printf("%s", json);

    free(data);
    free(blocks);
    return receipt.pass ? 0 : 1;
}
