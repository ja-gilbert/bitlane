"""Tests per second for each CPU simulator at several test counts, same stimulus.

Every number covers simulation only: the C reference is timed around its one
call with the stimulus already packed, Icarus between its "loaded" and
"simulated" markers, and Verilator inside its own loop. File I/O never counts.
"""

import time
from pathlib import Path

from bitlane import native
from bitlane.icarus import run_icarus
from bitlane.levels import Packed
from bitlane.netlist import Netlist
from bitlane.stimulus import random_inputs
from bitlane.verilator import run_verilator


def benchmark(
    verilog: Path,
    top: str,
    netlist: Netlist,
    packed: Packed,
    reset: str | None,
    cycles: int,
    sizes: list[int],
    threads: int,
    repeats: int = 3,
) -> list[dict]:
    """One row per test count with the best-of-`repeats` seconds of each simulator;
    Icarus runs once, since it is the slow one."""
    probe_net = native.probe_nets(netlist)
    icarus_dir = Path("build") / f"{top}_icarus"
    verilator_dir = Path("build") / "verilator" / top
    rows = []
    for n_tests in sizes:
        inputs = random_inputs(netlist, cycles, n_tests, reset)
        stim_net, stim_words = native.pack_stimulus(packed, inputs)
        seconds = []
        for _ in range(repeats):
            start = time.perf_counter()
            native.run(packed, stim_net, stim_words, probe_net)
            seconds.append(time.perf_counter() - start)
        row = {"tests": n_tests, "cycles": cycles, "C reference": min(seconds)}
        row["Icarus"] = run_icarus(verilog, top, netlist, inputs, icarus_dir)[2]
        for n in (1, threads):
            runs = [
                run_verilator(verilog, top, netlist, inputs, verilator_dir, n)[1]
                for _ in range(repeats)
            ]
            row[f"Verilator, {n} thread{'s' if n > 1 else ''}"] = min(runs)
        rows.append(row)
    return rows


def table(rows: list[dict], top: str, cpu: str) -> str:
    """The rows as a Markdown table of tests per second, with a caption line."""
    columns = [key for key in rows[0] if key not in ("tests", "cycles")]
    lines = [
        f"{top}: tests per second, {rows[0]['cycles']} cycles per test, simulation only, {cpu}",
        "",
        "| tests | " + " | ".join(columns) + " |",
        "|---:|" + "---:|" * len(columns),
    ]
    for row in rows:
        cells = [f"{row['tests'] / row[c]:,.0f}" for c in columns]
        lines.append(f"| {row['tests']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cpu_name() -> str:
    """The CPU model from /proc/cpuinfo, for the table's caption."""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown CPU"
