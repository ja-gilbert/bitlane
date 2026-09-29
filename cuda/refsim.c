// The C reference simulator: the same packed-bit algorithm as refsim.py, in plain C.
// It is the CPU baseline of the benchmark and the code the CUDA kernel is ported from.
//
// Bit t of vals[net * n_words + w] is the net's value in test 32*w + t.
// Net 0 is constant 0 and net 1 is constant 1. Gates arrive sorted by level.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { NOT, AND, OR, XOR, MUX };  // the order of KINDS in levels.py

// Runs n_cycles cycles. stim_words[cycle][i] is the packed row of input net stim_net[i];
// probe_words[cycle][i] receives the packed row of output net probe_net[i].
void bitlane_simulate(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net, const uint32_t *stim_words,
    int n_probe, const int32_t *probe_net, uint32_t *probe_words)
{
    size_t row = (size_t)n_words * sizeof(uint32_t);  // bytes in one net's words
    uint32_t *vals = calloc((size_t)n_nets * n_words, sizeof(uint32_t));
    uint32_t *next = malloc((size_t)n_flops * row);  // what every flop will capture
    memset(vals + n_words, 0xFF, row);  // net 1 is constant 1

    for (int cycle = 0; cycle < n_cycles; cycle++) {
        for (int i = 0; i < n_stim; i++) {  // apply the inputs
            memcpy(vals + (size_t)stim_net[i] * n_words, stim_words, row);
            stim_words += n_words;
        }

        for (int level = 0; level < n_levels; level++)  // one kernel launch per level
            for (int g = level_start[level]; g < level_start[level + 1]; g++) {
                const uint32_t *a = vals + (size_t)in_nets[3 * g] * n_words;
                const uint32_t *b = vals + (size_t)in_nets[3 * g + 1] * n_words;
                const uint32_t *s = vals + (size_t)in_nets[3 * g + 2] * n_words;
                uint32_t *y = vals + (size_t)out_net[g] * n_words;
                // One statement per word of 32 tests. On the GPU each (gate, word) is a
                // thread that runs this switch once. Here the switch sits outside the word
                // loop so the compiler can vectorise each loop: 2 to 4 times faster.
                switch (kind[g]) {
                case NOT: for (int w = 0; w < n_words; w++) y[w] = ~a[w]; break;
                case AND: for (int w = 0; w < n_words; w++) y[w] = a[w] & b[w]; break;
                case OR:  for (int w = 0; w < n_words; w++) y[w] = a[w] | b[w]; break;
                case XOR: for (int w = 0; w < n_words; w++) y[w] = a[w] ^ b[w]; break;
                default:  // MUX: b where s is 1, a where s is 0
                    for (int w = 0; w < n_words; w++) y[w] = (b[w] & s[w]) | (a[w] & ~s[w]);
                }
            }

        for (int i = 0; i < n_probe; i++) {  // sample the outputs
            memcpy(probe_words, vals + (size_t)probe_net[i] * n_words, row);
            probe_words += n_words;
        }

        for (int f = 0; f < n_flops; f++)  // the clock edge: read every D first ...
            memcpy(next + (size_t)f * n_words, vals + (size_t)flop_d[f] * n_words, row);
        for (int f = 0; f < n_flops; f++)  // ... then write every Q
            memcpy(vals + (size_t)flop_q[f] * n_words, next + (size_t)f * n_words, row);
    }
    free(next);
    free(vals);
}
