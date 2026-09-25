"""bitlane: a GPU gate-level logic simulator. Command line entry point."""

import argparse
from pathlib import Path

from bitlane.synth import cell_counts, synth


def main() -> None:
    parser = argparse.ArgumentParser(prog="bitlane")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="flatten a Verilog design to a JSON netlist")
    p.add_argument("verilog", type=Path, help="Verilog source file")
    p.add_argument("--top", help="top module (default: the file's stem)")

    args = parser.parse_args()
    if args.command == "synth":
        top = args.top or args.verilog.stem
        out = Path("build") / f"{top}.json"
        synth(args.verilog, top, out)
        for cell_type, n in sorted(cell_counts(out).items()):
            print(f"{n:6}  {cell_type}")
        print(f"wrote {out}")
