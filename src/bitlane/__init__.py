"""bitlane: a GPU gate-level logic simulator. Command line entry point."""

import argparse
import time
from pathlib import Path

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
    p = sub.add_parser("check", help="compare the NumPy simulator with Icarus Verilog")
    p.add_argument("verilog", type=Path, help="Verilog source file")
    p.add_argument("--top", help="top module (default: the file's stem)")
    p.add_argument("--tests", type=int, default=10000)
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--reset", help="input port to hold high in cycle 0")

    args = parser.parse_args()
    try:
        run(args)
    except ValueError as e:  # bad netlist, loop, unknown port, mismatch: one line
        raise SystemExit(f"bitlane: {e}") from e


def run(args: argparse.Namespace) -> None:
    if args.command == "synth":
        top = args.top or args.verilog.stem
        out = Path("build") / f"{top}.json"
        synth(args.verilog, top, out)
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

    elif args.command == "check":
        top = args.top or args.verilog.stem
        out = Path("build") / f"{top}.json"
        synth(args.verilog, top, out)
        netlist = read_netlist(out)
        if netlist.clock and not args.reset:
            raise ValueError(
                "a clocked design needs --reset so every test starts from reset"
            )
        inputs = random_inputs(netlist, args.cycles, args.tests, args.reset)
        ours = simulate(pack(netlist), inputs)
        workdir = Path("build") / f"{top}_icarus"
        theirs, unknown = run_icarus(args.verilog, top, netlist, inputs, workdir)
        first_cycle = 1 if netlist.clock else 0  # the reset cycle is not compared
        if mismatch := first_mismatch(ours, theirs, unknown, first_cycle):
            raise ValueError(mismatch)
        print(f"{top}: {args.tests} tests x {args.cycles} cycles match Icarus")
