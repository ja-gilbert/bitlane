"""Run a design in Icarus Verilog on our stimulus and read back what it computed.

A generated testbench loads one row of input bits per (test, cycle) from a hex
file and, for every row, drives the inputs after the clock edge, samples the
outputs one time unit before the next edge, then makes the edge: the cycle model
of refsim. The tests run back to back in one simulation, so a test starts in
whatever state the previous one left (x for the very first), which is why the
compare starts after the reset cycle.
"""

import subprocess
from pathlib import Path

import numpy as np

from bitlane.netlist import Netlist

TESTBENCH = """\
`timescale 1ns/1ps
module tb;
    reg {clock} = 0;
{signals}
    {top} dut({connections});
    reg [{row_bits}:0] tb_stim [0:{last_row}];
    integer tb_i, tb_f;
    initial begin
        $readmemh("{stim}", tb_stim);
        tb_f = $fopen("{out}", "w");
        for (tb_i = 0; tb_i <= {last_row}; tb_i = tb_i + 1) begin
            #1 {{{inputs}}} = tb_stim[tb_i];  // after the edge
            #3 $fwrite(tb_f, "{formats}\\n", {outputs});  // one unit before the edge
            #1 {clock} = 1;
            #5 {clock} = 0;
        end
        $fclose(tb_f);
        $finish;
    end
endmodule
"""


def driven_inputs(netlist: Netlist) -> list[str]:
    """The input ports the stimulus drives: every input except the clock."""
    return [name for name in netlist.inputs if name != netlist.clock]


def write_stimulus(netlist: Netlist, inputs: dict, path: Path) -> None:
    """One hex row per (test, cycle), test-major: the driven inputs concatenated."""
    names = driven_inputs(netlist)
    widths = [len(netlist.inputs[name]) for name in names]
    columns = [inputs[name].T.reshape(-1) for name in names]  # test-major order
    with path.open("w") as f:
        for values in zip(*columns):
            row = 0
            for value, width in zip(values, widths):
                row = (row << width) | int(value)
            f.write(f"{row:x}\n")


def testbench(netlist: Netlist, top: str, n_rows: int, stim: Path, out: Path) -> str:
    """The Verilog testbench that replays `stim` through `top` and writes `out`."""
    ports = netlist.inputs | netlist.outputs
    inputs, outputs = driven_inputs(netlist), list(netlist.outputs)
    signals = [f"    reg [{len(ports[n]) - 1}:0] {n};" for n in inputs]
    signals += [f"    wire [{len(ports[n]) - 1}:0] {n};" for n in outputs]
    return TESTBENCH.format(
        clock=netlist.clock or "clk",
        signals="\n".join(signals),
        top=top,
        connections=", ".join(f".{n}({n})" for n in ports),
        row_bits=sum(len(ports[n]) for n in inputs) - 1,
        last_row=n_rows - 1,
        stim=stim,
        out=out,
        inputs=", ".join(inputs),
        formats=" ".join("%h" for _ in outputs),
        outputs=", ".join(outputs),
    )


def run_icarus(
    verilog: Path, top: str, netlist: Netlist, inputs: dict, workdir: Path
) -> tuple[dict, dict]:
    """Icarus's outputs for `inputs` (each (n_cycles, n_tests)) in the same shape,
    plus, per port, an (n_cycles, n_tests) mask of where Icarus printed x or z."""
    workdir = workdir.resolve()  # the testbench embeds the paths; keep them absolute
    workdir.mkdir(parents=True, exist_ok=True)
    n_cycles, n_tests = next(iter(inputs.values())).shape
    n_rows = n_cycles * n_tests
    stim = workdir / "stim.hex"
    out = workdir / "out.txt"
    tb = workdir / "tb.v"
    vvp = workdir / "tb.vvp"
    write_stimulus(netlist, inputs, stim)
    tb.write_text(testbench(netlist, top, n_rows, stim, out))
    subprocess.run(["iverilog", "-g2012", "-o", vvp, tb, verilog], check=True)
    subprocess.run(["vvp", "-n", vvp], check=True, stdout=subprocess.DEVNULL)

    values = {name: np.zeros(n_rows, dtype=np.uint64) for name in netlist.outputs}
    unknown = {name: np.zeros(n_rows, dtype=bool) for name in netlist.outputs}
    with out.open() as f:
        for row, line in enumerate(f):
            for name, field in zip(netlist.outputs, line.split()):
                # %h prints a digit as x, or X when only some of its bits are x.
                if "x" in field.lower() or "z" in field.lower():
                    unknown[name][row] = True
                else:
                    values[name][row] = int(field, 16)

    def per_cycle(flat: np.ndarray) -> np.ndarray:
        """Test-major rows -> (n_cycles, n_tests)."""
        return flat.reshape(n_tests, n_cycles).T

    outputs = {name: per_cycle(v) for name, v in values.items()}
    return outputs, {name: per_cycle(u) for name, u in unknown.items()}


def first_mismatch(
    ours: dict, theirs: dict, unknown: dict, first_cycle: int
) -> str | None:
    """Describe the first disagreement from first_cycle on, or return None."""
    for name, mine in ours.items():
        bad = (mine != theirs[name]) | unknown[name]
        bad[:first_cycle] = False
        if bad.any():
            cycle, test = np.argwhere(bad)[0]
            other = "x" if unknown[name][cycle, test] else theirs[name][cycle, test]
            return (
                f"{name} differs at test {test}, cycle {cycle}: "
                f"ours {mine[cycle, test]}, reference {other}"
            )
    return None
