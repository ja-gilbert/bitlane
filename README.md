# bitlane

bitlane is a gate-level logic simulator that runs on the GPU. Yosys flattens a Verilog
design into AND, OR, XOR, NOT and MUX gates plus D flip-flops, Python sorts the gates
into levels, and a CUDA kernel evaluates one level per launch for many tests at once.
Each 32-bit word holds the value of one net in 32 different tests, one test per bit, so
a single AND instruction simulates an AND gate 32 times. The simulator is checked
against Icarus Verilog, cycle by cycle, on random tests.

**The result.** On an 8-bit ALU (297 gates) the GPU simulates one million random tests
at 1.3 billion tests per second, counting the upload of the stimulus and the download
of the outputs. That is about 3 times Verilator on 24 cores and 4.5 times a vectorised
single-core C version of the same algorithm at the same batch size (2.0 times that C
version at its own best batch size). At 100,000 tests per run and below, and on designs
of one or two dozen gates, the CPU wins. The numbers, the reasons and what the profiler
says are below.

![Tests per second against tests per run for the mux, the ALU and the state machine](results/throughput.png)

## How it works

1. **Synthesis.** `scripts/synth.ys` has Yosys flatten the design into five gate types
   and one flip-flop type (positive edge, D input). Enables and synchronous resets
   become logic in front of D. The result is a JSON netlist.
2. **Levels.** Inputs, constants and flip-flop outputs are level 0. A gate's level is
   one more than the highest level among its inputs (Kahn's algorithm). Gates in the
   same level do not depend on each other, so they can all be evaluated at once.
   Gates left unsorted are on a combinational loop or fed by one, and are reported as
   an error.
3. **Packing.** The gates are stored as flat arrays sorted by level. Values live in
   `vals[row][word]`, one row per net: bit `t` of word `w` is the net's value in test
   `32 * w + t`. A gate is then one bitwise operation per word; a MUX is
   `(b & s) | (a & ~s)`. A gate's output takes over the row of a net that no later
   level reads, the way a register allocator or an ML compiler's memory planner reuses
   buffers, so the ALU's 318 nets need 148 rows.
4. **The kernel.** One launch per level, one thread per (gate, word). Thread `i` takes
   gate `i / n_words` and word `i % n_words`, so neighbouring threads read and write
   neighbouring words of the same rows and the memory accesses coalesce.
5. **The clock.** Each cycle applies the inputs, evaluates the levels in order, samples
   the outputs, then lets every flip-flop capture its D: read every D first, then
   write every Q.

| File | What it holds |
|---|---|
| `scripts/synth.ys`, `src/bitlane/synth.py` | the Yosys flow |
| `src/bitlane/netlist.py` | the JSON reader: nets, gates, flip-flops, the clock |
| `src/bitlane/levels.py` | level sorting, row assignment and the packed arrays |
| `src/bitlane/refsim.py` | the NumPy simulator, the readable version of the algorithm |
| `cuda/refsim.c` | the same algorithm in C: the CPU baseline |
| `cuda/kernel.cu` | the CUDA kernel |
| `src/bitlane/native.py` | calls the C and CUDA code (`cuda/bitlane.h`) through ctypes |
| `src/bitlane/stimulus.py` | random stimulus, with the reset port held at 1 in the first cycle |
| `src/bitlane/icarus.py` | the Icarus harness and the cycle-by-cycle compare |
| `src/bitlane/verilator.py` | the Verilator harness |
| `src/bitlane/bench.py`, `scripts/chart.py` | the benchmark and the chart |
| `scripts/gpu_run.py` | one GPU workload, for the profiler |
| `src/bitlane/__init__.py` | the `bitlane` command: synth, levels, sim, check, bench |

## Correctness

Icarus Verilog is the reference. A generated testbench replays the same random stimulus
through the original Verilog, and every output is compared in every cycle of every test.

bitlane is two-state (0 or 1) and its flip-flops start at 0 in every test. Icarus is
four-state, its flip-flops start at x, and it runs all the tests back to back in one
simulation, so each test starts where the previous one ended. So every test of a
clocked design begins with a reset cycle, and the comparison starts in the cycle after
it. An x or z from Icarus after reset counts as a mismatch.

```bash
uv run bitlane check designs/alu.v --gpu
uv run bitlane check designs/fsm.v --reset rst --gpu
uv run pytest
```

`check` runs 10,000 random tests of 20 cycles each through the NumPy simulator, or
through the CUDA kernel with `--gpu`. The test suite compares the NumPy simulator, the
C version and the CUDA kernel with Icarus on five designs, the kernel with the C
version word for word, and the Verilator harness with the NumPy simulator. It also
covers edge cases in the netlist reader (x and z bits, outputs wired straight to an
input or a constant, two clocks) and in the level sorter (a combinational loop).

The benchmark only times the simulators. It does not compare their outputs.

## Results

Millions of tests per second. Warm runs: set up once, then stimulus in host memory to
outputs in host memory, wall clock, median of 5. CPU: Intel Core i9-14900HX, 24 threads
for Verilator. GPU: NVIDIA GeForce RTX 4060 Laptop GPU, PCIe gen 4 x8.

**mux**: 8-bit 4:1 multiplexer, 24 gates in 2 levels, 1 cycle per test.

| tests | C reference | Icarus | Verilator, 1 thread | Verilator, 24 threads | GPU |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,027 | 0.652 | 37.1 | 139 | 17.3 |
| 10,000 | 2,716 | 0.686 | 37.4 | 442 | 187 |
| 100,000 | 3,163 | 0.685 | 37.3 | 473 | 559 |
| 1,000,000 | 1,528 | 0.685 | 35.1 | 535 | 1,427 |

**alu**: 8-bit ALU with eight operations, 297 gates in 17 levels, 1 cycle per test.

| tests | C reference | Icarus | Verilator, 1 thread | Verilator, 24 threads | GPU |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 361 | 0.458 | 34.8 | 85.8 | 5.66 |
| 10,000 | 679 | 0.474 | 35.1 | 360 | 67.8 |
| 100,000 | 642 | 0.469 | 35.3 | 356 | 407 |
| 1,000,000 | 299 | 0.470 | 33.5 | 420 | 1,336 |

**fsm**: sequence detector, 14 gates in 5 levels and 2 flip-flops, 20 cycles per test.

| tests | C reference | Icarus | Verilator, 1 thread | Verilator, 24 threads | GPU |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 346 | 0.0279 | 0.837 | 8.53 | 0.762 |
| 10,000 | 854 | 0.0272 | 0.855 | 10.9 | 4.74 |
| 100,000 | 566 | 0.0276 | 0.862 | 9.71 | 59.1 |
| 1,000,000 | 376 | 0.0276 | 0.864 | 11.7 | 343 |

Cold runs (open + run + close, so the buffers are allocated and freed inside the timed
call) at 1,000,000 tests, in millions of tests per second:

| design | C reference, cold | its allocation | GPU, cold | its allocation |
|---|---:|---:|---:|---:|
| mux | 284 | 2.87 ms | 266 | 3.05 ms |
| alu | 100 | 6.70 ms | 246 | 3.31 ms |
| fsm | 242 | 1.47 ms | 106 | 6.50 ms |

On the ALU the GPU's allocation falls under 10% of the total time after 40 million
tests through one open simulator.

## When the GPU wins and when it doesn't

**It wins with many tests and enough gates.** The ALU at one million tests is the one
case here: 1,336 million tests per second against 420 million for Verilator on 24 cores
and 299 million for the C reference.

**The C reference's best is at a smaller batch.** Its speed follows the cache sizes. On
the ALU, 10,000 and 100,000 tests need 0.2 and 1.8 MiB of net values, inside one core's
2 MiB L2 cache, and run at 679 and 642 million tests per second. One million tests need
18 MiB, which spills to the 36 MiB L3 cache, and run at 299 million. Cache misses were
not counted, so the cause is an inference. A fair summary is that the GPU at its best
is 2.0 times one CPU core at its best, and 4.5 times at the same batch size.

**It loses at 100,000 tests per run and below.** A run of the ALU is 19 kernel launches
and two copies across PCIe however few tests there are. The GPU takes between 0.15 and
0.25 ms for anything from 1,000 to 100,000 ALU tests: the time is fixed cost, not work.
A launch costs about 7 µs: at 1,000 tests the state machine's 181 launches take 1.3 ms.
The crossover with the C reference is between 100,000 and one million tests.

**It loses on tiny designs.** The mux has 24 gates in 2 levels. The C reference stays
ahead at every size and the GPU only comes within 7% at one million tests. There the GPU
takes 0.70 ms on the mux and 0.75 ms on the ALU, which has 12 times the gates, because
most of the mux's run is not gate evaluation: in an instrumented run of the earlier
version, 0.38 ms of its 0.75 ms was the upload and 0.20 ms the download, 5.25 MB across
PCIe in pageable memory, and 0.16 ms the kernels.

**It loses on tiny designs with a clock too.** Twenty cycles of the state machine are
181 launches for 14 gates. The GPU reaches 343 million tests per second at one million
tests and the C reference 376 million.

**A cold call pays for setup.** Opening the simulator, running one million ALU tests
once and closing it gives the GPU 246 million tests per second instead of 1,336
million. The C reference drops too, from 299 million to 100 million, so cold the GPU is
still 2.5 times ahead on the ALU, but behind on the mux and the state machine. Either
one pays off when an open simulator is reused.

**The algorithm matters more than the hardware.** The single-core C reference beats
Verilator on 24 cores on the mux and the state machine at every size, by 2.9 times and
by 32 times at one million tests. Verilator simulates one test at a time; packing
32 tests into a word, which the compiler then vectorises, is what makes both the C
version and the GPU version fast.

## What the profiler says

Nsight Compute on the ALU at one million tests, the 17 `eval_level` launches of one
run, with the SM clock locked to its base of 1.88 GHz (so each kernel takes longer here
than at the GPU's normal clocks; the memory clock read 7 to 8 GHz from launch to
launch). The profile uses application replay and no cache flush, so each launch sees
the caches roughly as a real run leaves them. The profiler's default is to flush the
caches before every launch, which made the kernels 70 to 80% slower in total and sent
most of their reads to DRAM (a 58% L2 hit rate instead of 98%); its kernel replay,
which restores memory between passes, lost the layout change's effect altogether.

| | before | after |
|---|---:|---:|
| rows of net values | 318 (38 MiB) | 148 (18 MiB) |
| the 17 launches, total | 323 µs | 285 µs |
| warp instructions | 26.5 M | 26.5 M |
| DRAM read + written | 16 + 30 MB | 4 + 32 MB |
| L2 hit rate | 98% | 92% |
| SM throughput, the 8 levels over 1,800 blocks | 33 to 62% of peak | 49 to 63% |

"After" is the row reuse described under How it works. It was meant to cut DRAM
traffic by making the working set fit the 32 MB L2 cache, and the profile shows that
the L2 was already serving nearly every read before the change: the DRAM traffic is
mostly the write-back of the stored rows (297 rows of 125 kB is 37 MB). The reads that
did go to DRAM fell from 16 to 4 MB, and the launches got 12% faster on the same
instructions, which the counters do not pin down: the hit rate is the one figure that
moved between repeated profiles (98 to 100% before the change, 92 to 100% after),
while the times, the instructions and the written bytes held. A second pair of
profiles through a C driver gave 268 to 236 µs, the same 12%; the absolute times move
with the memory clock, which the profiler does not lock. An interleaved wall-clock
timing of the same phase, 40 runs each, agrees (0.433 to 0.379 ms). The C reference
gained 31% at one million tests, probably because its 18 MiB now fit the CPU's L3
cache. At four million tests levels 2 to 8 run at 79 to 93% of peak DRAM throughput at
the profiler's memory clock, with a 74% L2 hit rate, before and after alike: there the
kernel is bound by DRAM bandwidth, and the live rows (74 MB) no longer fit the L2
whatever the layout.

Two more things the profile settled. The data loads coalesce: a warp's 32 words are one
request, and at four million tests that request is four 32-byte sectors. At one million
a row is 31,250 words, not a multiple of 8, so most requests straddle five sectors and
the loads cost 12% more sectors than they need; rounding the row length up to a
multiple of 8 words would fix it. And at one million tests no unit of the GPU is
saturated: the eight big levels keep the issue slots 61 to 67% busy and DRAM at 45 to
71% of its peak, over 13 to 48 full waves each, so neither bandwidth nor the short
launches explains the time; memory latency is the likely remainder.

The kernel-only ceiling: everything between the end of the upload and the end of the
last kernel, 19 launches and the gaps between them, takes 0.38 ms per million ALU tests
at the GPU's normal clocks, about 2.6 billion tests per second if the copies were free.
The kernels themselves add up to 0.26 to 0.31 ms in the profiles, at a lower clock than
normal, so at least a fifth and probably a third of that is launch and synchronisation
overhead, which is why fewer, larger launches is the next step. The state machine's 181 launches of a few microseconds each
are the extreme case. The table's 1.3 billion includes the copies.

```bash
ncu --set full --replay-mode application --cache-control none -k regex:eval_level \
    --launch-skip 34 --launch-count 17 -o alu uv run scripts/gpu_run.py alu 1000000 1 - 3
```

"Before" is the commit before row reuse; the command profiles the third of the three
runs.

## What each column includes

Every simulator gets the same random stimulus at each size. Every figure is wall-clock
time, the median of five runs. File I/O is never timed for any simulator.

- **C reference.** One call of `bitlane_run` on an open simulator: copy the stimulus
  rows in, evaluate, copy the output rows out. One thread, `gcc -O3 -march=native`.
- **GPU.** One call of `bitlane_run_gpu` on an open simulator: upload the stimulus,
  launch every level of every cycle, download the outputs. The buffers on the GPU are
  allocated beforehand. The download waits for every queued launch, so the timer stops
  only when the results are in host memory.
- **Packing.** The C reference and the GPU are given the stimulus already packed 32
  tests to a word and leave the outputs packed. The packing is done once in NumPy,
  untimed, and the benchmark does not unpack the outputs. Icarus and Verilator read one
  row per test and cycle inside their timed loops.
- **Icarus.** The testbench loads the stimulus file before the first cycle and writes
  the outputs after the last. It prints a marker at each of those two points and the
  time between the markers is the figure.
- **Verilator.** The model and the loop around it are compiled with
  `-O3 -march=native`. The stimulus is in memory and the timer is around the simulation
  loop alone. With 24 threads, each thread has its own model instance and simulates
  its share of the tests; the threads are started, and each model has run once, before
  the timer starts. Verilator's own `--threads` option parallelises within one model,
  which suits large designs and is not the fair baseline for running independent tests
  in parallel, so it is not in the table.
- **Before timing.** The C reference and the GPU run the same workload untimed for 1.5
  seconds first. Before that the GPU is brought to its full clock with one second of
  memory fills. This laptop's GPU drops to a low clock after a few idle seconds, and a
  light workload (few tests, or a small design) leaves it at about a fifth of its full
  clock, up to ten times slower, for as long as it runs. The wake-up is there so that
  every GPU figure is taken at the full clock; the clock itself is not recorded.
- **Cold.** Open, one run and close, timed together: allocating the buffers, uploading
  the netlist and freeing everything, per call. The process has already created its
  CUDA context and loaded the kernels, so GPU start-up is not included.

24 is the number of physical cores (8 performance, 16 efficiency). The 24-thread
Verilator figure is the noisiest in the tables: forty runs of the ALU at one million
tests ranged from 175 to 523 million tests per second, with a median of 437. The
slowest of the 24 threads decides the time.

## What transfers to ML kernels

The kernel is small, but several of the questions it raised come up in ML kernels too.

- **Layout decides whether accesses coalesce.** `vals[row][word]` with one thread per
  word puts neighbouring threads on neighbouring addresses. It is the same choice as
  which dimension of a tensor is contiguous. The profiler added the alignment lesson:
  at one million tests a row of 125,000 bytes is not a multiple of 32, and the loads
  cost 12% more memory sectors than at four million; padding the row length, as
  libraries pad a tensor's leading dimension, would fix it.
- **A level is an elementwise kernel, not a matrix multiply.** Each thread reads two
  words (three for a MUX) and writes one for a single bitwise operation: the low
  arithmetic intensity of an activation function, not of a dense layer. Frameworks fuse
  such kernels into their neighbours, and fusing levels would be the same move.
- **Buffers are reused by lifetime.** Row reuse is the memory planning an ML compiler
  does for a graph's intermediate tensors: two values share memory when their lifetimes
  do not overlap. Here it halved the footprint and sped up the CPU version by 31%, but
  did not cut the GPU's DRAM traffic, which was its purpose.
- **Launch overhead sets the smallest useful batch.** The ALU's 19 launches cost the
  same whatever the batch size. Small-batch inference has the same problem and the
  same fixes: fusing kernels, and replaying the launch sequence with CUDA Graphs.
- **Transfers and allocation are part of the cost.** The headline number includes the
  upload and the download, and the cold number shows what allocating per call costs.
  Frameworks keep memory pools for the same reason.
- **Launches are asynchronous, so a timer needs a synchronisation point.** Timing the
  launch loop alone would measure how fast work is queued.
- **A measurement has to control the machine's state.** The same kernel ran up to ten
  times slower when the GPU had idled for a few seconds beforehand, and 70% slower
  under the profiler's default cache flush. Profiled three ways, the layout change
  measured 8% with the flush, nothing with kernel replay and no flush, and 12% with
  application replay and no flush; only the last matched the wall clock.
- **A fast kernel is checked against a slow, trusted one.** Icarus here; a CPU
  reference implementation for a custom operator.

## Limits

- One clock, positive edge only, no memories, no latches. An asynchronous reset is
  rejected by the Yosys flow.
- The reset must be synchronous and active-high: `--reset` holds the named port at 1
  in the first cycle, and that cycle must leave every flip-flop at a known value.
- Two-state only: an x in the netlist becomes 0 with a warning, a z is an error.
- Ports are at most 64 bits wide.
- A design whose only input is the clock is not supported.
- The stimulus is uniform random. After the first cycle the reset port is random like
  every other input, so tests of a clocked design seldom get far from reset.
- The designs are tiny, 14 to 297 gates. Nothing here says how a large design behaves.
- The C reference is single-threaded. The GPU at its best is only 2.0 times one core
  at its best batch size, so a multi-threaded C reference would very likely overtake
  the GPU on the ALU. It was not measured.
- The largest run is one million tests. The GPU's curve is still rising there on all
  three designs: from 100,000 to one million tests it gains 2.6 times on the mux, 3.3
  times on the ALU and 5.8 times on the state machine.
- The copies use pageable host memory and do not overlap the kernels.
- One machine, one laptop GPU. The numbers move with temperature and background load.
- One kernel launch per level, nothing fused.

## Future work

- Fewer launches: fuse small levels, or replay the launch sequence with CUDA Graphs.
- Pinned host memory, and copies that overlap the kernels.
- 64-bit words.
- A multi-threaded C reference.
- A real design: PicoRV32, which needs memories.

## Build and run

Needs an NVIDIA GPU with the CUDA toolkit, CMake 3.24 or newer, gcc, g++ and make
(`build-essential` on Ubuntu), Yosys, Icarus Verilog, Verilator and
[uv](https://docs.astral.sh/uv/). Developed on Ubuntu 24.04 under WSL2 with CUDA 13.3,
Yosys 0.33, Icarus Verilog 12.0 and Verilator 5.020. The shared library is built by
CMake the first time it is needed. `cuda/CMakeLists.txt` targets the RTX 4060's
architecture (`sm_89`); change `CUDA_ARCHITECTURES` for another GPU.

```bash
uv run pytest                              # ours against Icarus, Verilator against ours
uv run bitlane synth designs/alu.v         # Verilog to gates: build/alu.json
uv run bitlane levels build/alu.json       # gates per level, nets and rows
uv run bitlane check designs/alu.v --gpu   # 10,000 random tests against Icarus
```

The three benchmark runs behind the tables, and the chart. `--threads` is the number
of physical cores; each run writes `results/bench_<design>.json`.

```bash
uv run bitlane bench designs/mux.v --threads 24
uv run bitlane bench designs/alu.v --threads 24
uv run bitlane bench designs/fsm.v --reset rst --cycles 20 --threads 24
uv run scripts/chart.py
```

## License

MIT.
