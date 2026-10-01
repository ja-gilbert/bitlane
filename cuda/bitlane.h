// libbitlane.so: the packed-bit simulator twice, with the same three calls each.
// refsim.c is the C reference (bitlane_open, bitlane_run, bitlane_close) and kernel.cu
// is the CUDA kernel (the same names ending in _gpu). Python calls them through ctypes
// (src/bitlane/native.py).
//
// open  copies the netlist and allocates every buffer, once.
// run   simulates n_cycles cycles from reset and can be called again and again: a warm
//       run. stim_words[cycle][i] is the packed row of input net stim_net[i], and
//       probe_words[cycle][i] receives the packed row of output net probe_net[i].
// close frees everything.
//
// Nets hold one uint32 word per 32 tests; net 0 is constant 0 and net 1 constant 1;
// gates arrive sorted by level. Every cycle: apply the stimulus rows, evaluate the gates
// level by level, copy the probe rows out, then let every flop capture its D (read all
// D before writing any Q). Every run starts with all flops at 0.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitlane_sim bitlane_sim;          // an open C reference simulator
typedef struct bitlane_gpu_sim bitlane_gpu_sim;  // an open GPU simulator

bitlane_sim *bitlane_open(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net,
    int n_probe, const int32_t *probe_net);
void bitlane_run(bitlane_sim *sim, const uint32_t *stim_words, uint32_t *probe_words);
void bitlane_close(bitlane_sim *sim);

bitlane_gpu_sim *bitlane_open_gpu(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net,
    int n_probe, const int32_t *probe_net);
void bitlane_run_gpu(bitlane_gpu_sim *sim, const uint32_t *stim_words, uint32_t *probe_words);
void bitlane_close_gpu(bitlane_gpu_sim *sim);

#ifdef __cplusplus
}
#endif
