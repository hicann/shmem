/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef BARRIER_RELAY_KERNEL_H
#define BARRIER_RELAY_KERNEL_H

void relay_put_barrier_test_do(void *stream, uint64_t config, uint8_t *slots, int rank_id, int rank_size,
    int rounds);
void barrier_perf_v3_do(void *stream, uint64_t config, int iters);
void barrier_perf_relay_do(void *stream, uint64_t config, int iters);
void relay_put_barrier_perf_do(void *stream, uint64_t config, uint8_t *slots, int rank_id, int rank_size, int iters);

#endif // BARRIER_RELAY_KERNEL_H
