"""Each simulator against Icarus Verilog on every design, same stimulus."""

import sys
from pathlib import Path

import pytest
from helpers import DESIGNS, RESET, ROOT, load

from bitlane import refsim
from bitlane.icarus import first_mismatch, run_icarus
from bitlane.levels import pack
from bitlane.netlist import read_netlist
from bitlane.stimulus import random_inputs
from bitlane.synth import SCRIPT, synth


@pytest.mark.parametrize("name", DESIGNS)
def test_matches_icarus(name, tmp_path, simulate):
    netlist, _ = load(name)
    reset = RESET.get(name)
    inputs = random_inputs(netlist, n_cycles=20, n_tests=200, reset=reset)
    ours = simulate(pack(netlist), inputs)
    theirs, unknown, _ = run_icarus(
        ROOT / "designs" / f"{name}.v", name, netlist, inputs, tmp_path
    )
    # A clocked design starts each test in the previous test's state (x for the first),
    # so its reset cycle is not compared; a combinational one is compared everywhere.
    assert first_mismatch(ours, theirs, unknown, first_cycle=1 if reset else 0) is None


def icarus_mismatch(verilog: Path, top: str, workdir: Path) -> str | None:
    """Synthesize `top`, run 100 random tests through refsim and Icarus, compare."""
    synth(verilog, top, workdir / f"{top}.json")
    netlist = read_netlist(workdir / f"{top}.json")
    inputs = random_inputs(netlist, n_cycles=1, n_tests=100)
    theirs, unknown, _ = run_icarus(verilog, top, netlist, inputs, workdir)
    ours = refsim.simulate(pack(netlist), inputs)
    return first_mismatch(ours, theirs, unknown, first_cycle=0)


def test_names_cannot_clash_with_the_testbench(tmp_path):
    # Each once broke the harness: a top module called tb, the testbench's own name,
    # and ports called clk, tb_clk or tb_stim, names of nets the testbench declared.
    verilog = tmp_path / "design.v"  # not tb.v: run_icarus writes its testbench there
    verilog.write_text(
        "module tb(input wire clk, input wire tb_clk, input wire tb_stim,\n"
        "          output wire y);\n"
        "    assign y = clk ^ tb_clk ^ tb_stim;\n"
        "endmodule\n"
    )
    assert icarus_mismatch(verilog, "tb", tmp_path) is None


def test_paths_may_contain_spaces(tmp_path, monkeypatch):
    # The design, the netlist, the testbench files and the synthesis script, which
    # lives in the repository, all in a directory with a space in its name.
    workdir = tmp_path / "with space"
    workdir.mkdir()
    script = workdir / "synth.ys"
    script.write_text(SCRIPT.read_text())
    # Through sys.modules: the package's attribute bitlane.synth is the function.
    monkeypatch.setattr(sys.modules["bitlane.synth"], "SCRIPT", script)
    verilog = workdir / "comb.v"
    verilog.write_text(
        "module comb(input wire a, input wire b, output wire y);\n"
        "    assign y = a & b;\n"
        "endmodule\n"
    )
    assert icarus_mismatch(verilog, "comb", workdir) is None
