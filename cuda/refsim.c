// The C reference simulator: the same packed-bit algorithm as refsim.py, in plain C.
// It is the CPU baseline of the benchmark and the code the CUDA kernel is ported from.
//
// Bit t of vals[net * n_words + w] is the net's value in test 32*w + t.
// Net 0 is constant 0 and net 1 is constant 1. Gates arrive sorted by level.

#include <stdlib.h>
#include <string.h>

#include "bitlane.h"

enum { NOT, AND, OR, XOR, MUX };  // the order of KINDS in levels.py

struct bitlane_sim {
    int n_words, n_cycles, n_levels, n_flops, n_stim, n_probe;
    int32_t *level_start, *in_nets, *out_net, *flop_d, *flop_q, *stim_net, *probe_net;
    uint8_t *kind;
    uint32_t *vals;  // one row of n_words per net
    uint32_t *next;  // one row per flop: what it will capture at the edge
};

// A private copy of `count` elements of `size` bytes, so the caller may free its own.
static void *copy_of(const void *source, size_t count, size_t size) {
    void *copy = malloc(count * size + 1);  // + 1: never ask for zero bytes
    memcpy(copy, source, count * size);
    return copy;
}

// See bitlane.h for what the arguments mean.
bitlane_sim *bitlane_open(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net,
    int n_probe, const int32_t *probe_net)
{
    size_t n_gates = level_start[n_levels];
    bitlane_sim *sim = malloc(sizeof *sim);
    sim->n_words = n_words, sim->n_cycles = n_cycles, sim->n_levels = n_levels;
    sim->n_flops = n_flops, sim->n_stim = n_stim, sim->n_probe = n_probe;
    sim->level_start = copy_of(level_start, n_levels + 1, sizeof(int32_t));
    sim->kind = copy_of(kind, n_gates, sizeof(uint8_t));
    sim->in_nets = copy_of(in_nets, 3 * n_gates, sizeof(int32_t));
    sim->out_net = copy_of(out_net, n_gates, sizeof(int32_t));
    sim->flop_d = copy_of(flop_d, n_flops, sizeof(int32_t));
    sim->flop_q = copy_of(flop_q, n_flops, sizeof(int32_t));
    sim->stim_net = copy_of(stim_net, n_stim, sizeof(int32_t));
    sim->probe_net = copy_of(probe_net, n_probe, sizeof(int32_t));
    sim->vals = calloc((size_t)n_nets * n_words + 1, sizeof(uint32_t));
    sim->next = malloc(((size_t)n_flops * n_words + 1) * sizeof(uint32_t));
    memset(sim->vals + n_words, 0xFF, n_words * sizeof(uint32_t));  // net 1 is constant 1
    return sim;
}

void bitlane_run(bitlane_sim *sim, const uint32_t *stim_words, uint32_t *probe_words) {
    int n_words = sim->n_words;
    size_t row = (size_t)n_words * sizeof(uint32_t);  // bytes in one net's words
    uint32_t *vals = sim->vals, *next = sim->next;
    const int32_t *in_nets = sim->in_nets;

    // Every run starts from reset. Only the flop outputs carry state from one run to the
    // next: every other net is an input, a constant, or recomputed before it is read.
    for (int f = 0; f < sim->n_flops; f++)
        memset(vals + (size_t)sim->flop_q[f] * n_words, 0, row);

    for (int cycle = 0; cycle < sim->n_cycles; cycle++) {
        for (int i = 0; i < sim->n_stim; i++) {  // apply the inputs
            memcpy(vals + (size_t)sim->stim_net[i] * n_words, stim_words, row);
            stim_words += n_words;
        }

        for (int level = 0; level < sim->n_levels; level++)  // one kernel launch per level
            for (int g = sim->level_start[level]; g < sim->level_start[level + 1]; g++) {
                const uint32_t *a = vals + (size_t)in_nets[3 * g] * n_words;
                const uint32_t *b = vals + (size_t)in_nets[3 * g + 1] * n_words;
                const uint32_t *s = vals + (size_t)in_nets[3 * g + 2] * n_words;
                uint32_t *y = vals + (size_t)sim->out_net[g] * n_words;
                // One statement per word of 32 tests. On the GPU each (gate, word) is a
                // thread that runs this switch once. Here the switch sits outside the word
                // loop so the compiler can vectorise each loop: 2 to 4 times faster.
                switch (sim->kind[g]) {
                case NOT: for (int w = 0; w < n_words; w++) y[w] = ~a[w]; break;
                case AND: for (int w = 0; w < n_words; w++) y[w] = a[w] & b[w]; break;
                case OR:  for (int w = 0; w < n_words; w++) y[w] = a[w] | b[w]; break;
                case XOR: for (int w = 0; w < n_words; w++) y[w] = a[w] ^ b[w]; break;
                default:  // MUX: b where s is 1, a where s is 0
                    for (int w = 0; w < n_words; w++) y[w] = (b[w] & s[w]) | (a[w] & ~s[w]);
                }
            }

        for (int i = 0; i < sim->n_probe; i++) {  // sample the outputs
            memcpy(probe_words, vals + (size_t)sim->probe_net[i] * n_words, row);
            probe_words += n_words;
        }

        for (int f = 0; f < sim->n_flops; f++)  // the clock edge: read every D first ...
            memcpy(next + (size_t)f * n_words, vals + (size_t)sim->flop_d[f] * n_words, row);
        for (int f = 0; f < sim->n_flops; f++)  // ... then write every Q
            memcpy(vals + (size_t)sim->flop_q[f] * n_words, next + (size_t)f * n_words, row);
    }
}

void bitlane_close(bitlane_sim *sim) {
    void *buffers[] = {sim->level_start, sim->kind, sim->in_nets, sim->out_net, sim->flop_d,
                       sim->flop_q, sim->stim_net, sim->probe_net, sim->vals, sim->next, sim};
    for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++) free(buffers[i]);
}
