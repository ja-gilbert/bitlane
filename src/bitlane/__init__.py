"""bitlane: a GPU gate-level logic simulator. Command line entry point."""

import argparse
from pathlib import Path

from bitlane.levels import pack
from bitlane.netlist import read_netlist
from bitlane.synth import cell_counts, synth


def main() -> None:
    parser = argparse.ArgumentParser(prog="bitlane")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="flatten a Verilog design to a JSON netlist")
    p.add_argument("verilog", type=Path, help="Verilog source file")
    p.add_argument("--top", help="top module (default: the file's stem)")

    p = sub.add_parser("levels", help="sort a JSON netlist into levels and print sizes")
    p.add_argument("netlist", type=Path, help="JSON netlist written by bitlane synth")

    args = parser.parse_args()
    if args.command == "synth":
        top = args.top or args.verilog.stem
        out = Path("build") / f"{top}.json"
        synth(args.verilog, top, out)
        for cell_type, n in sorted(cell_counts(out).items()):
            print(f"{n:6}  {cell_type}")
        print(f"wrote {out}")
    elif args.command == "levels":
        try:
            netlist = read_netlist(args.netlist)
            packed = pack(netlist)
        except ValueError as e:
            raise SystemExit(f"bitlane: {e}") from e
        starts = packed.level_start
        depth = len(starts) - 1
        for k in range(depth):
            print(f"level {k + 1:3}: {starts[k + 1] - starts[k]:5} gates")
        print(
            f"{len(netlist.gates)} gates, {len(netlist.flops)} flops, "
            f"{netlist.n_nets} nets, depth {depth}"
        )
