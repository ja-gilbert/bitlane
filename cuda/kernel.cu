// CUDA kernel v1: refsim.c with its two inner loops turned into a grid of threads.
// One kernel launch per level, one thread per (gate, word of 32 tests). All state
// stays on the GPU: open allocates every buffer once, and each run sends the stimulus
// up in one copy before the first cycle and brings the outputs back in one after the last.

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "bitlane.h"

enum { NOT, AND, OR, XOR, MUX };  // the order of KINDS in levels.py
static const int BLOCK = 256;     // threads per block

// One thread: gate `first + i / n_words` of this level, on word `i % n_words`.
// Consecutive threads take consecutive words of the same gate, so each of the three
// reads and the write walks along one row of vals: the accesses coalesce.
__global__ void eval_level(uint32_t *vals, const uint8_t *kind, const int32_t *in_nets,
                           const int32_t *out_net, int first, int n_gates, int n_words) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n_gates * n_words) return;  // the last block may overhang
    int g = first + (int)(i / n_words), w = (int)(i % n_words);
    uint32_t a = vals[(size_t)in_nets[3 * g] * n_words + w];
    uint32_t b = vals[(size_t)in_nets[3 * g + 1] * n_words + w];
    uint32_t s = vals[(size_t)in_nets[3 * g + 2] * n_words + w];
    uint32_t y;
    switch (kind[g]) {
    case NOT: y = ~a; break;
    case AND: y = a & b; break;
    case OR:  y = a | b; break;
    case XOR: y = a ^ b; break;
    default:  y = (b & s) | (a & ~s);  // MUX: b where s is 1, a where s is 0
    }
    vals[(size_t)out_net[g] * n_words + w] = y;
}

// Row r of dst (or row dst_row[r]) = row r of src (or row src_row[r]); one thread per
// word. Applies the stimulus, samples the probes and does both halves of the clock edge.
__global__ void copy_rows(uint32_t *dst, const int32_t *dst_row, const uint32_t *src,
                          const int32_t *src_row, int n_rows, int n_words) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n_rows * n_words) return;
    int r = (int)(i / n_words), w = (int)(i % n_words);
    size_t from = (size_t)(src_row ? src_row[r] : r) * n_words + w;
    size_t to = (size_t)(dst_row ? dst_row[r] : r) * n_words + w;
    dst[to] = src[from];
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "bitlane: CUDA error: %s\n", cudaGetErrorString(status));
        exit(1);
    }
}

// `count` elements on the GPU, copied up from `host` unless it is null.
template <typename T> static T *on_gpu(const T *host, size_t count) {
    if (count == 0) return nullptr;
    T *device;
    check(cudaMalloc(&device, count * sizeof(T)));
    if (host) check(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

static unsigned blocks(size_t threads) { return (unsigned)((threads + BLOCK - 1) / BLOCK); }

static void rows(uint32_t *dst, const int32_t *dst_row, const uint32_t *src,
                 const int32_t *src_row, int n_rows, int n_words) {
    if (n_rows > 0)
        copy_rows<<<blocks((size_t)n_rows * n_words), BLOCK>>>(dst, dst_row, src, src_row,
                                                              n_rows, n_words);
}

struct bitlane_gpu_sim {
    int n_words, n_cycles, n_levels, n_flops, n_stim, n_probe;
    int32_t *level_start;  // on the host: the launch loop reads it
    uint8_t *kind;         // everything below lives on the GPU
    int32_t *in_nets, *out_net, *flop_d, *flop_q, *stim_net, *probe_net;
    uint32_t *vals;   // one row of n_words per net
    uint32_t *next;   // one row per flop: what it will capture at the edge
    uint32_t *stim;   // the stimulus rows of every cycle
    uint32_t *probe;  // the probed output rows of every cycle
};

// See bitlane.h for what the arguments mean.
bitlane_gpu_sim *bitlane_open_gpu(
    int n_nets, int n_words, int n_cycles,
    int n_levels, const int32_t *level_start,
    const uint8_t *kind, const int32_t *in_nets, const int32_t *out_net,
    int n_flops, const int32_t *flop_d, const int32_t *flop_q,
    int n_stim, const int32_t *stim_net,
    int n_probe, const int32_t *probe_net)
{
    size_t n_gates = level_start[n_levels];
    bitlane_gpu_sim *sim = new bitlane_gpu_sim;
    sim->n_words = n_words, sim->n_cycles = n_cycles, sim->n_levels = n_levels;
    sim->n_flops = n_flops, sim->n_stim = n_stim, sim->n_probe = n_probe;
    sim->level_start = new int32_t[n_levels + 1];
    for (int i = 0; i <= n_levels; i++) sim->level_start[i] = level_start[i];
    sim->kind = on_gpu(kind, n_gates);
    sim->in_nets = on_gpu(in_nets, 3 * n_gates), sim->out_net = on_gpu(out_net, n_gates);
    sim->flop_d = on_gpu(flop_d, n_flops), sim->flop_q = on_gpu(flop_q, n_flops);
    sim->stim_net = on_gpu(stim_net, n_stim), sim->probe_net = on_gpu(probe_net, n_probe);
    sim->vals = on_gpu<uint32_t>(nullptr, (size_t)n_nets * n_words);
    sim->next = on_gpu<uint32_t>(nullptr, (size_t)n_flops * n_words);
    sim->stim = on_gpu<uint32_t>(nullptr, (size_t)n_cycles * n_stim * n_words);
    sim->probe = on_gpu<uint32_t>(nullptr, (size_t)n_cycles * n_probe * n_words);
    check(cudaMemset(sim->vals, 0, (size_t)n_nets * n_words * sizeof(uint32_t)));
    check(cudaMemset(sim->vals + n_words, 0xFF, n_words * sizeof(uint32_t)));  // net 1 is 1
    check(cudaDeviceSynchronize());  // open's queued work is finished, and counted, here
    return sim;
}

void bitlane_run_gpu(bitlane_gpu_sim *sim, const uint32_t *stim_words, uint32_t *probe_words) {
    int n_words = sim->n_words, n_flops = sim->n_flops;
    if (n_words == 0) return;  // no tests: nothing to launch
    size_t stim_cycle = (size_t)sim->n_stim * n_words, probe_cycle = (size_t)sim->n_probe * n_words;
    uint32_t *vals = sim->vals;

    // The stimulus of every cycle goes up in one copy.
    check(cudaMemcpy(sim->stim, stim_words, sim->n_cycles * stim_cycle * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));
    // Every run starts from reset. Only the flop outputs carry state from one run to the
    // next: every other net is an input, a constant, or recomputed before it is read.
    check(cudaMemset(sim->next, 0, (size_t)n_flops * n_words * sizeof(uint32_t)));
    rows(vals, sim->flop_q, sim->next, nullptr, n_flops, n_words);

    for (int cycle = 0; cycle < sim->n_cycles; cycle++) {
        rows(vals, sim->stim_net, sim->stim + cycle * stim_cycle, nullptr, sim->n_stim, n_words);
        for (int level = 0; level < sim->n_levels; level++) {  // one launch per level
            int first = sim->level_start[level], n = sim->level_start[level + 1] - first;
            eval_level<<<blocks((size_t)n * n_words), BLOCK>>>(vals, sim->kind, sim->in_nets,
                                                               sim->out_net, first, n, n_words);
        }
        rows(sim->probe + cycle * probe_cycle, nullptr, vals, sim->probe_net, sim->n_probe, n_words);
        rows(sim->next, nullptr, vals, sim->flop_d, n_flops, n_words);  // the clock edge: read every D,
        rows(vals, sim->flop_q, sim->next, nullptr, n_flops, n_words);  // then write every Q
    }

    // The launches above are only queued. This copy waits for all of them and reports a
    // kernel that failed while running; a launch that never started shows up in the next line.
    check(cudaMemcpy(probe_words, sim->probe, sim->n_cycles * probe_cycle * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    check(cudaGetLastError());
}

void bitlane_close_gpu(bitlane_gpu_sim *sim) {
    void *buffers[] = {sim->kind, sim->in_nets, sim->out_net, sim->flop_d, sim->flop_q,
                       sim->stim_net, sim->probe_net, sim->vals, sim->next, sim->stim, sim->probe};
    for (void *buffer : buffers) check(cudaFree(buffer));
    delete[] sim->level_start;
    delete sim;
}

// This laptop's GPU drops to a low clock after a few idle seconds, and a light workload
// does not bring it back up. This keeps it busy for `milliseconds`, filling a 256 MB
// buffer over and over, so that a benchmark can start every timed run at the full clock.
// It waits for each fill so that the deadline counts finished fills, not queued ones.
void bitlane_wake_gpu(int milliseconds) {
    const size_t bytes = 256 << 20;
    void *buffer;
    check(cudaMalloc(&buffer, bytes));
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (std::chrono::steady_clock::now() < end) {
        check(cudaMemset(buffer, 0, bytes));
        check(cudaDeviceSynchronize());
    }
    check(cudaFree(buffer));
}
