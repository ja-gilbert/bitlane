// The two entry points of libbitlane.so. Same arguments, same meaning, same results:
// bitlane_simulate is the C reference in refsim.c; bitlane_simulate_gpu is the CUDA
// kernel in kernel.cu. Python calls both through ctypes (src/bitlane/native.py).
//
// Runs n_cycles cycles of the packed-bit simulation. Nets hold one uint32 word per
// 32 tests; net 0 is constant 0 and net 1 constant 1; gates arrive sorted by level.
// stim_words[cycle][i] is the packed row of input net stim_net[i], and
// probe_words[cycle][i] receives the packed row of output net probe_net[i].
// Every cycle: apply the stimulus rows, evaluate the gates level by level, copy the
// probe rows out, then let every flop capture its D (read all D before writing any Q).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void bitlane_simulate(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net, const uint32_t *stim_words,
    int n_probe, const int32_t *probe_net, uint32_t *probe_words);

void bitlane_simulate_gpu(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net, const uint32_t *stim_words,
    int n_probe, const int32_t *probe_net, uint32_t *probe_words);

#ifdef __cplusplus
}
#endif
