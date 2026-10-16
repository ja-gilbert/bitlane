# What the profiler says

Profiling the ALU at one million tests with Nsight Compute showed that the kernel is
not bound by DRAM bandwidth at that size: the GPU's 32 MB L2 cache serves most of the
reads, no unit of the GPU is saturated, and at least a sixth of the kernel phase is
launch overhead. The memory-layout change made for this profile, row reuse (a gate's
output takes over the row of a net whose last reader is in an earlier level), cut the
working set from 38 to 18 MiB and took 12% off the launches, but not for the reason it
was meant to: the L2 was already serving most of the reads, the write-back of the
output rows stays, and the 12% held in profiles that read almost nothing from DRAM
before or after. At four million tests the big levels are bound by DRAM bandwidth, and
the layout no longer matters.

**Method.** Nsight Compute on the ALU at one million tests, the 17 `eval_level`
launches of one run, with the SM clock held at about 1.88 GHz by the profiler's clock
control (so each kernel takes longer here than at the GPU's normal clocks; the memory
clock read about 7 GHz). The profile uses application replay and no cache flush, so
each launch sees the caches roughly as a real run leaves them. The profiler's default
is to flush the caches before every launch, which made the kernels 42 to 48% slower in
total than in the table below, sent 51 MB of their reads to DRAM both before and after
the change instead of 16 and 4 MB, and shrank the layout change's effect to 8%; its
kernel replay, which restores memory between passes, lost the effect altogether.

| | before | after |
|---|---:|---:|
| rows of net values | 318 (38 MiB) | 148 (18 MiB) |
| the 17 launches, total | 323 µs | 285 µs |
| warp instructions | 26.5 M | 26.5 M |
| DRAM read + written | 16 + 30 MB | 4 + 32 MB |
| SM throughput, the 8 levels over 1,800 blocks | 33 to 62% of peak | 49 to 63% |

"After" is row reuse. It was meant to cut DRAM traffic by making the working set fit
the 32 MB L2 cache. The profile shows that the L2 was already serving most of the reads
before the change: of the 84 MB or so that the launches read, 16 MB came from DRAM, and
4 MB after. Most of the DRAM traffic is the write-back of the stored rows (297 rows of
125 kB is 37 MB), which row reuse does not remove. The launches took 12% less time on
the same instructions, which the counters do not pin down.

A second pair of profiles, through a small standalone program that runs the same CUDA
code without Python (not in the repository), read almost nothing from DRAM before or
after (0.5 and 0.0 MB) and still gave 268 to 236 µs, the same 12%; between the two
pairs the DRAM reads, the written bytes and the absolute times all moved, and only the
instruction count and the 12% held. An interleaved wall-clock timing of the same phase,
40 runs each, agrees (0.433 to 0.379 ms).

The C reference became 31% faster at one million tests in the two benchmark runs made
for this comparison (228 to 299 million tests per second; the later run in
[the benchmark](benchmark.md) gave 290), probably because its 18 MiB now fit the CPU's
L3 cache.

At four million tests levels 2 to 8 run at 79 to 93% of peak DRAM throughput at the
profiler's memory clock, with a 63% L2 hit rate, before and after alike: there the
kernel is bound by DRAM bandwidth, and the live rows (74 MB) no longer fit the L2
whatever the layout.

Two more things the profile settled. The data loads coalesce: a warp's 32 words are one
request, and at four million tests that request is four 32-byte sectors. At one million
a row is 31,250 words, not a multiple of 8, so three requests in four straddle five
sectors and the data loads cost 19% more sectors than they need; rounding the row
length up to a multiple of 8 words would fix it. And at one million tests no unit of
the GPU is saturated: after row reuse the eight big levels keep the issue slots 61 to
67% busy and DRAM at 45 to 71% of its peak, over 13 to 48 waves each, so neither
bandwidth nor the short launches explains the time; memory latency is the likely
remainder.

The kernel-only ceiling: everything between the end of the upload and the end of the
last kernel, 19 launches and the gaps between them, takes 0.38 ms per million ALU tests
at the GPU's normal clocks, about 2.6 billion tests per second if the copies were free.
The 19 kernels themselves (the 17 levels and the two row copies) add up to 0.26 to
0.31 ms in the profiles, at a lower clock than normal, so at least a sixth and probably
a third of that is launch and synchronisation overhead, which is why fewer, larger
launches is the next step. The state machine's 181 launches of a few microseconds each
are the extreme case. The benchmark's 1.3 billion includes the copies.

## Reproducing

```bash
ncu --set full --replay-mode application --cache-control none -k regex:eval_level \
    --launch-skip 34 --launch-count 17 -o alu uv run scripts/gpu_run.py alu 1000000 1 - 3
```

The command profiles the third of the three runs. "Before" is commit e792c08, the last
one before row reuse; it has no `scripts/gpu_run.py`, so copy this one in and change
`native.probe_nets(packed)` to `native.probe_nets(netlist)`. Nsight Compute needs
access to the GPU's performance counters; on Windows, with WSL2, that is a setting in
the NVIDIA Control Panel.
