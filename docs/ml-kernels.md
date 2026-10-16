# What transfers to ML kernels

The kernel is small, but several of the questions it raised come up in ML kernels too.
The numbers below are from [the benchmark](benchmark.md) and [the profile](profiling.md).

- **Layout decides whether accesses coalesce.** `vals[row][word]` with one thread per
  word puts neighbouring threads on neighbouring addresses. It is the same choice as
  which dimension of a tensor is contiguous. The profiler added the alignment lesson:
  at one million tests a row of 125,000 bytes is not a multiple of 32, and the data
  loads cost 19% more memory sectors per request than at four million; padding the row
  length, as libraries pad a tensor's leading dimension, would fix it.
- **A level is an elementwise kernel, not a matrix multiply.** Each thread loads two
  or three words and stores one for a single bitwise operation: the low arithmetic
  intensity of an activation function, not of a dense layer. Frameworks fuse such
  kernels into their neighbours, and fusing levels would be the same move.
- **Buffers are reused by lifetime.** Row reuse is the memory planning an ML compiler
  does for a graph's intermediate tensors: two values share memory when their lifetimes
  do not overlap. Here it halved the footprint and made the CPU version 31% faster at
  one million tests. On the GPU it was meant to cut DRAM traffic, but the L2 was
  already serving most reads: DRAM reads fell from 16 to 4 MB, the write-back stayed,
  and the launches took 12% less time for reasons the counters do not pin down.
- **Launch overhead sets the smallest useful batch.** The ALU's 19 launches cost the
  same whatever the batch size. Small-batch inference has the same problem and the
  same fixes: fusing kernels, and replaying the launch sequence with CUDA Graphs.
- **Transfers, allocation and data preparation are part of the cost.** The headline
  number includes the upload and the download, the cold number shows what allocating
  per call costs, and the packing that the benchmark leaves out costs more than the run
  it feeds. Frameworks keep memory pools, and keep data in the layout the kernel wants,
  for the same reasons.
- **Launches are asynchronous, so a timer needs a synchronisation point.** Timing the
  launch loop alone would measure how fast work is queued.
- **A measurement has to control the machine's state.** The same kernel ran up to ten
  times slower when the GPU had idled for a few seconds beforehand, and over 40%
  slower under the profiler's default cache flush. Profiled three ways, the layout
  change measured 8% with the flush, nothing with kernel replay and no flush, and 12%
  with application replay and no flush; only the last matched the wall clock.
- **A fast kernel is checked against a slow, trusted one.** Icarus here; a CPU
  reference implementation for a custom operator.
