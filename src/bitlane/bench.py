"""Tests per second for every simulator at several test counts, on the same stimulus.

Warm runs are the headline: set up first, then timed from stimulus in host memory to
outputs in host memory, so the GPU pays for its upload and download. The C reference
and the GPU are also timed cold: open + run + close.

Before timing, the workload runs untimed for SETTLE seconds to pay the first-run
costs and let the CPU's clock settle, and the GPU is woken with WAKE_MS of memory
fills, because an idle GPU drops to a low clock that a light workload does not bring
back. The README section "What each column includes" has the full method.
"""

import ctypes
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from bitlane import native
from bitlane.icarus import run_icarus
from bitlane.levels import Packed
from bitlane.netlist import Netlist
from bitlane.stimulus import random_inputs
from bitlane.verilator import run_verilator

SETTLE = 1.5  # seconds of untimed runs before a timed measurement
WAKE_MS = 1000  # milliseconds of memory fills that bring the GPU to its full clock


def median_seconds(action: Callable[[], object], repeats: int) -> float:
    """The median wall-clock time of `repeats` calls of `action`."""
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds)


def time_native(
    packed: Packed, stim_net, stim_words, probe_net, gpu: bool, repeats: int
) -> tuple[float, float]:
    """(warm, cold) seconds of the C reference or the GPU on this stimulus."""
    n_cycles, _, n_words = stim_words.shape

    def cold() -> None:
        session = native.open_session(
            packed, stim_net, probe_net, n_cycles, n_words, gpu
        )
        native.run(session, stim_words)
        native.close_session(session)

    session = native.open_session(packed, stim_net, probe_net, n_cycles, n_words, gpu)
    # The timed call goes straight to the library with its pointers worked out once, so
    # the microseconds of checks and conversions in native.run are not timed, just as
    # Verilator's timer inside its own process leaves out everything around its loop.
    run = native.library()["bitlane_run" + session.suffix]  # a fresh function object
    run.restype, run.argtypes = None, [ctypes.c_void_p] * 3
    pointers = session.sim, stim_words.ctypes.data, session.probe_words.ctypes.data
    if gpu:
        native.library().bitlane_wake_gpu(WAKE_MS)
    # Not timed: this open created the CUDA context, the first run loads the kernels and
    # touches the buffers, and an idle GPU or CPU needs time to come back up to speed.
    settled = time.perf_counter() + SETTLE
    while time.perf_counter() < settled:
        run(*pointers)
    warm = median_seconds(lambda: run(*pointers), repeats)
    native.close_session(session)
    return warm, median_seconds(cold, repeats)


def benchmark(
    verilog: Path,
    top: str,
    netlist: Netlist,
    packed: Packed,
    reset: str | None,
    cycles: int,
    sizes: list[int],
    threads: int,
    repeats: int = 5,
) -> list[dict]:
    """One row per test count: the warm seconds of every simulator, and under "cold"
    the open + run + close seconds of the C reference and the GPU."""
    # Pin glibc's mmap threshold at its default. Freeing a big buffer otherwise raises
    # it (up to 32 MB), later opens then reuse the freed memory, and their cold times
    # skip the page faults that a first open pays.
    ctypes.CDLL(None).mallopt(-3, 128 * 1024)  # -3 is M_MMAP_THRESHOLD
    probe_net = native.probe_nets(packed)
    icarus_dir = Path("build") / f"{top}_icarus"
    verilator_dir = Path("build") / "verilator" / top
    rows = []
    for n_tests in sizes:
        inputs = random_inputs(netlist, cycles, n_tests, reset)
        stim_net, stim_words = native.pack_stimulus(packed, inputs)
        c_warm, c_cold = time_native(
            packed, stim_net, stim_words, probe_net, False, repeats
        )
        gpu_warm, gpu_cold = time_native(
            packed, stim_net, stim_words, probe_net, True, repeats
        )
        row = {"tests": n_tests, "cycles": cycles, "C reference": c_warm}
        icarus = run_icarus(verilog, top, netlist, inputs, icarus_dir, repeats)
        row["Icarus"] = icarus[2]  # its seconds
        for n in (1, threads):
            runs = [
                run_verilator(verilog, top, netlist, inputs, verilator_dir, n)[1]
                for _ in range(repeats)
            ]
            label = f"Verilator, {n} thread{'s' if n > 1 else ''}"
            row[label] = statistics.median(runs)
        row["GPU"] = gpu_warm
        row["cold"] = {"C reference": c_cold, "GPU": gpu_cold}
        rows.append(row)
    return rows


def millions(tests: int, seconds: float) -> str:
    """Millions of tests per second: whole millions from 99.5 up, three significant
    figures below. Timings repeat no better than that."""
    rate = tests / seconds / 1e6
    return f"{rate:,.0f}" if rate >= 99.5 else f"{rate:#.3g}"  # '#' keeps a final 0


def table(rows: list[dict], top: str, machine: str, repeats: int) -> str:
    """The warm table in Markdown with its caption, then the cold lines of the largest run."""
    columns = [key for key in rows[0] if key not in ("tests", "cycles", "cold")]
    cycles = rows[0]["cycles"]
    per_test = "1 cycle" if cycles == 1 else f"{cycles} cycles"
    caption = (
        f"{top}: millions of tests per second, {per_test} per test. "
        f"Warm runs: set up once, then stimulus in host memory to outputs in host memory, "
        f"wall clock, median of {repeats}. {machine}"
    )
    lines = [
        caption,
        "",
        "| tests | " + " | ".join(columns) + " |",
        "|---:|" + "---:|" * len(columns),
    ]
    for row in rows:
        cells = [millions(row["tests"], row[c]) for c in columns]
        lines.append(f"| {row['tests']:,} | " + " | ".join(cells) + " |")
    last = rows[-1]
    lines.append("")
    for name, cold in last["cold"].items():
        warm = last[name]
        allocation = cold - warm
        # Allocation is paid once per open simulator: it is under a tenth of the total
        # time once the tests run through it take nine times as long as it does.
        amortized = 9 * allocation * last["tests"] / warm
        lines.append(
            f"{name}, cold (open + run + close) at {last['tests']:,} tests: "
            f"{millions(last['tests'], cold)} million tests per second. Allocation costs "
            f"{allocation * 1e3:.2f} ms and falls under 10% of the total after "
            f"{amortized:,.0f} tests through one open simulator."
        )
    return "\n".join(lines)


def machine_name(threads: int) -> str:
    """CPU model, Verilator thread count, GPU model and PCIe link, for the caption."""
    cpu = "unknown CPU"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip().replace("(R)", "").replace("(TM)", "")
            break
    query = "name,pcie.link.gen.current,pcie.link.width.current"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]
    gpu, gen, width = (
        subprocess.run(command, check=True, capture_output=True, text=True)
        .stdout.strip()
        .split(", ")
    )
    return f"CPU: {cpu}, {threads} threads for Verilator. GPU: {gpu}, PCIe gen {gen} x{width}."
