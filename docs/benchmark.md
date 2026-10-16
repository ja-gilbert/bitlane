# The benchmark

Four simulators, Verilator in two configurations, on the same random stimulus at four
batch sizes: the C reference (`cuda/refsim.c`, the packed algorithm on one CPU core),
Icarus Verilog, Verilator on one thread and on 24 threads, and the CUDA kernel. Every
figure is wall-clock time, the median of five runs, in millions of tests per second.
CPU: Intel Core i9-14900HX, 24 Verilator worker threads. GPU: NVIDIA GeForce RTX 4060
Laptop GPU, PCIe gen 4 x8.

## Results

Warm runs: set up once, then stimulus in host memory to outputs in host memory.

**mux**: 8-bit 4:1 multiplexer, 24 gates in 2 levels, 1 cycle per test.

| tests | C reference | Icarus | Verilator, 1 thread | Verilator, 24 threads | GPU |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,316 | 0.797 | 38.8 | 51.0 | 19.7 |
| 10,000 | 3,033 | 0.799 | 38.3 | 364 | 187 |
| 100,000 | 3,259 | 0.827 | 38.2 | 430 | 617 |
| 1,000,000 | 1,739 | 0.821 | 37.2 | 483 | 1,399 |

**alu**: 8-bit ALU with eight operations, 297 gates in 17 levels, 1 cycle per test.

| tests | C reference | Icarus | Verilator, 1 thread | Verilator, 24 threads | GPU |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 252 | 0.512 | 35.1 | 19.1 | 5.22 |
| 10,000 | 715 | 0.537 | 35.7 | 233 | 59.0 |
| 100,000 | 659 | 0.544 | 36.4 | 204 | 422 |
| 1,000,000 | 290 | 0.533 | 34.7 | 343 | 1,334 |

**fsm**: sequence detector, 14 gates in 5 levels and 2 flip-flops, 20 cycles per test.

| tests | C reference | Icarus | Verilator, 1 thread | Verilator, 24 threads | GPU |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 414 | 0.0284 | 0.891 | 10.3 | 0.811 |
| 10,000 | 865 | 0.0291 | 0.898 | 7.54 | 8.23 |
| 100,000 | 595 | 0.0282 | 0.874 | 5.85 | 52.8 |
| 1,000,000 | 413 | 0.0274 | 0.871 | 12.7 | 364 |

Cold runs (open + run + close, so the buffers are allocated and freed inside the timed
call) at 1,000,000 tests, with what the cold call adds to the warm run:

| design | C reference, cold | cold-run overhead | GPU, cold | cold-run overhead |
|---|---:|---:|---:|---:|
| mux | 305 | 2.70 ms | 279 | 2.87 ms |
| alu | 104 | 6.18 ms | 256 | 3.16 ms |
| fsm | 234 | 1.86 ms | 143 | 4.24 ms |

On the ALU the GPU's cold-run overhead falls under 10% of the total time after 38
million tests through one open simulator.

The benchmark only times the simulators. The GPU was checked separately against Icarus
on each of these one-million-test runs' exact stimulus (`bitlane check --gpu --tests
1000000`, with each design's cycle count and reset), and matched on all three.

## When the GPU wins and when it doesn't

**It wins with many tests and enough gates.** The ALU at one million tests is the one
case here: 1,334 million tests per second against 343 million for Verilator on 24
threads and 290 million for the C reference.

**The C reference's best is at a smaller batch.** Its speed follows the cache sizes. On
the ALU, 10,000 and 100,000 tests need 0.2 and 1.8 MiB of net values, inside one core's
2 MiB L2 cache, and run at 715 and 659 million tests per second. One million tests need
18 MiB, which spills to the 36 MiB L3 cache, and run at 290 million. Cache misses were
not counted, so the cause is an inference. A fair summary is that the GPU at its best
is 1.9 times one CPU core at its best, and 4.6 times at the same batch size.

**It loses to the C reference at 100,000 tests per run and below.** A run of the ALU is
19 kernel launches and two copies across PCIe however few tests there are. The GPU takes
between 0.17 and 0.24 ms for anything from 1,000 to 100,000 ALU tests: the time is fixed
cost, not work. A launch costs about 7 µs: at 1,000 tests the state machine's 181
launches take 1.2 ms. On the ALU the crossover with the C reference is between 100,000
and one million tests.

**It loses on tiny designs.** The mux has 24 gates in 2 levels. The C reference stays
ahead at every size and the GPU is still 20% behind at one million tests. There the
GPU takes 0.71 ms on the mux and 0.75 ms on the ALU, which has 12 times the gates,
because most of the mux's run is not gate evaluation: in an instrumented mux run of
0.75 ms, 0.38 ms was the upload and 0.20 ms the download, 5.25 MB across PCIe in
pageable memory, and 0.16 ms the kernels.

**It loses on tiny designs with a clock too.** Twenty cycles of the state machine are
181 launches for 14 gates. The GPU reaches 364 million tests per second at one million
tests and the C reference 413 million.

**A cold call pays for setup.** Opening the simulator, running one million ALU tests
once and closing it gives the GPU 256 million tests per second instead of 1,334
million. The C reference drops too, from 290 million to 104 million, so cold the GPU is
still 2.5 times ahead on the ALU, but behind on the mux and the state machine. Either
one pays off when an open simulator is reused.

**The algorithm matters more than the hardware.** The single-core C reference beats
Verilator on 24 threads on the mux and the state machine at every size, by 3.6 times
and by 32 times at one million tests. Verilator simulates one test at a time; packing
32 tests into a word, which the compiler then vectorises, is what makes both the C
version and the GPU version fast.

**The GPU's curve is still rising at one million tests**, on all three designs: from
100,000 to one million tests it gains 2.3 times on the mux, 3.2 times on the ALU and
6.9 times on the state machine. Only the profile went further: at four million ALU
tests the big levels are bound by DRAM bandwidth ([the profile](profiling.md)).

## What each column includes

Every simulator gets the same random stimulus at each size. Every figure is wall-clock
time, the median of five runs. File I/O is never timed for any simulator.

- **C reference.** One call of `bitlane_run` on an open simulator: copy the stimulus
  rows in, evaluate, copy the output rows out. One thread, `gcc -O3 -march=native`.
- **GPU.** One call of `bitlane_run_gpu` on an open simulator: upload the stimulus,
  launch every level of every cycle, download the outputs. The buffers on the GPU are
  allocated beforehand. The download waits for every queued launch, so the timer stops
  only when the results are in host memory. The copies use pageable host memory and do
  not overlap the kernels.
- **Packing.** The C reference and the GPU are given the stimulus already packed 32
  tests to a word and leave the outputs packed. The packing is done once in NumPy,
  untimed, and the benchmark does not unpack the outputs. For a million ALU tests the
  packing takes about 106 ms and the unpacking 30 ms, against 0.75 ms for the GPU run
  itself. Icarus and Verilator read one row per test and cycle inside their timed loops.
- **Icarus.** The testbench loads the stimulus file before the first cycle and writes
  the outputs after the last. It prints a marker at each of those two points and the
  time between the markers is the figure.
- **Verilator.** The model and the loop around it are compiled with
  `-O3 -march=native`. The stimulus is in memory and the timer is around the simulation
  loop alone. With 24 threads, each thread has its own model instance and simulates
  its share of the tests; the threads are started, and each model has run once, before
  the timer starts. Verilator's own `--threads` option parallelises within one model,
  which suits large designs and is not the fair baseline for running independent tests
  in parallel, so it is not in the tables.
- **Before timing.** The C reference and the GPU run the same workload untimed for 1.5
  seconds first. Before that the GPU is brought to its full clock with one second of
  memory fills. This laptop's GPU drops to a low clock after a few idle seconds, and a
  light workload (few tests, or a small design) leaves it at about a fifth of its full
  clock, up to ten times slower, for as long as it runs. The wake-up is there so that
  every GPU figure is taken at the full clock; the clock itself is not recorded.
- **Cold.** Open, one run and close, timed together: allocating the buffers, uploading
  the netlist and freeing everything, per call. The process has already created its
  CUDA context and loaded the kernels, so GPU start-up is not included.

The CPU has 24 physical cores (8 performance, 16 efficiency), so the 24-thread run
starts one worker per core, but the threads are not pinned and the machine runs under
WSL2, so nothing guarantees that 24 cores were busy at once. That column is the
noisiest in the tables: forty runs of the ALU at one million tests ranged from 175 to
523 million tests per second with a median of 437, and the full benchmark's medians
have landed between 343 and 508 million. The slowest of the 24 threads decides the
time.

## Reproducing

`--threads` is the number of physical cores. Each run writes
`results/bench_<design>.json`; the chart is drawn from those files.

```bash
uv run bitlane bench designs/mux.v --threads 24
uv run bitlane bench designs/alu.v --threads 24
uv run bitlane bench designs/fsm.v --reset rst --cycles 20 --threads 24
uv run scripts/chart.py
```

Run them on an otherwise idle machine. The numbers move with temperature and
background load, and a run of the GPU at a small size is sensitive to the GPU's power
state despite the wake-up.
