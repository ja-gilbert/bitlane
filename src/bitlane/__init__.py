"""bitlane: a GPU gate-level logic simulator. Command line entry point."""

import argparse
import json
import time
from pathlib import Path

from bitlane import native
from bitlane.bench import benchmark, machine_name, table
from bitlane.icarus import first_mismatch, run_icarus
from bitlane.levels import pack
from bitlane.netlist import read_netlist
from bitlane.refsim import simulate
from bitlane.stimulus import random_inputs
from bitlane.synth import cell_counts, synth


def main() -> None:
    parser = argparse.ArgumentParser(prog="bitlane")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="flatten a Verilog design to a JSON netlist")
    p.add_argument("verilog", type=Path, help="Verilog source file")
    p.add_argument("--top", help="top module (default: the file's stem)")

    p = sub.add_parser("levels", help="sort a JSON netlist into levels and print sizes")
    p.add_argument("netlist", type=Path, help="JSON netlist written by bitlane synth")

    p = sub.add_parser("sim", help="run random tests through the NumPy simulator")
    p.add_argument("netlist", type=Path, help="JSON netlist written by bitlane synth")
    p.add_argument("--tests", type=int, default=1024)
    p.add_argument("--cycles", type=int, default=100)
    p.add_argument("--reset", help="input port to hold high in cycle 0")

    p = sub.add_parser("check", help="compare a simulator with Icarus Verilog")
    p.add_argument("verilog", type=Path, help="Verilog source file")
    p.add_argument("--top", help="top module (default: the file's stem)")
    p.add_argument("--tests", type=int, default=10000)
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--reset", help="input port to hold high in cycle 0")
    p.add_argument("--gpu", action="store_true", help="check the CUDA kernel")

    p = sub.add_parser("bench", help="tests per second for every simulator")
    p.add_argument("verilog", type=Path, help="Verilog source file")
    p.add_argument("--top", help="top module (default: the file's stem)")
    p.add_argument("--reset", help="input port to hold high in cycle 0")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--sizes", default="1000,10000,100000,1000000", help="test counts")
    p.add_argument("--threads", type=int, required=True, help="physical cores")
    p.add_argument("--repeats", type=int, default=5, help="runs per median")

    args = parser.parse_args()
    try:
        run(args)
    except ValueError as e:  # bad netlist, loop, unknown port, mismatch: one line
        raise SystemExit(f"bitlane: {e}") from e


def synthesize(args: argparse.Namespace) -> tuple[str, Path]:
    """Synthesize args.verilog to build/<top>.json; the top name and that path."""
    top = args.top or args.verilog.stem
    out = Path("build") / f"{top}.json"
    synth(args.verilog, top, out)
    return top, out


def run(args: argparse.Namespace) -> None:
    if args.command == "synth":
        _, out = synthesize(args)
        for cell_type, n in sorted(cell_counts(out).items()):
            print(f"{n:6}  {cell_type}")
        print(f"wrote {out}")
    elif args.command == "levels":
        netlist = read_netlist(args.netlist)
        packed = pack(netlist)
        starts = packed.level_start
        depth = len(starts) - 1
        for k in range(depth):
            print(f"level {k + 1:3}: {starts[k + 1] - starts[k]:5} gates")
        print(
            f"{len(netlist.gates)} gates, {len(netlist.flops)} flops, "
            f"{netlist.n_nets} nets, depth {depth}"
        )
    elif args.command == "sim":
        netlist = read_netlist(args.netlist)
        inputs = random_inputs(netlist, args.cycles, args.tests, args.reset)
        packed = pack(netlist)
        start = time.perf_counter()  # time the simulation only
        simulate(packed, inputs)
        seconds = time.perf_counter() - start
        rate = args.tests * args.cycles / seconds
        print(f"{args.tests} tests x {args.cycles} cycles: {rate:,.0f} test-cycles/s")

    elif args.command in ("check", "bench"):
        top, out = synthesize(args)
        netlist = read_netlist(out)
        if netlist.clock and not args.reset:
            raise ValueError(
                "a clocked design needs --reset so every test starts from reset"
            )
        if args.command == "bench":
            sizes = [int(size) for size in args.sizes.split(",")]
            rows = benchmark(
                args.verilog, top, netlist, pack(netlist), args.reset,
                args.cycles, sizes, args.threads, args.repeats,
            )  # fmt: skip
            machine = machine_name(args.threads)
            print(table(rows, top, machine, args.repeats))
            results = Path("results")
            results.mkdir(exist_ok=True)
            report = {"design": top, "machine": machine, "repeats": args.repeats}
            report["rows"] = rows
            (results / f"bench_{top}.json").write_text(json.dumps(report, indent=1))
            return
        inputs = random_inputs(netlist, args.cycles, args.tests, args.reset)
        if args.gpu:
            ours = native.simulate(pack(netlist), inputs, gpu=True)
            simulator = "the CUDA kernel"
        else:
            ours = simulate(pack(netlist), inputs)
            simulator = "the NumPy simulator"
        workdir = Path("build") / f"{top}_icarus"
        theirs, unknown, _ = run_icarus(args.verilog, top, netlist, inputs, workdir)
        first_cycle = 1 if netlist.clock else 0  # the reset cycle is not compared
        if mismatch := first_mismatch(ours, theirs, unknown, first_cycle):
            raise ValueError(mismatch)
        print(
            f"{top}: {simulator} matches Icarus on "
            f"{args.tests} tests x {args.cycles} cycles"
        )
