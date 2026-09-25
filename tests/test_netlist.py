"""The reader on the real designs (synthesized at test time) and a hand-made JSON."""

import json
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from bitlane.netlist import Netlist, read_netlist
from bitlane.synth import cell_counts, synth

ROOT = Path(__file__).parents[1]


def widths(ports: dict[str, list[int]]) -> dict[str, int]:
    return {name: len(bits) for name, bits in ports.items()}


@cache
def load(name: str) -> tuple[Netlist, Path]:
    """Synthesize designs/<name>.v with Yosys (once per run) and read it back."""
    path = ROOT / "build" / f"{name}.json"
    synth(ROOT / "designs" / f"{name}.v", name, path)
    return read_netlist(path), path


@pytest.mark.parametrize("name", ["counter", "adder"])
def test_cells_match_yosys(name):
    netlist, path = load(name)
    seen = Counter(f"$_{gate.kind}_" for gate in netlist.gates)
    seen["$_DFF_P_"] += len(netlist.flops)
    assert seen == cell_counts(path)


def test_counter_ports_and_clock():
    counter, _ = load("counter")
    assert widths(counter.inputs) == {"clk": 1, "rst": 1, "en": 1}
    assert widths(counter.outputs) == {"count": 8}
    assert counter.clock == "clk"
    assert len(counter.flops) == 8


def test_adder_is_combinational():
    adder, _ = load("adder")
    assert adder.flops == [] and adder.clock is None
    assert widths(adder.inputs) == {"a": 8, "b": 8, "cin": 1}
    assert widths(adder.outputs) == {"sum": 8, "cout": 1}


def test_unknown_cell_type(tmp_path):
    module = {"ports": {}, "cells": {"g1": {"type": "$_NAND_", "connections": {}}}}
    path = tmp_path / "nand.json"
    path.write_text(json.dumps({"modules": {"m": module}}))
    with pytest.raises(ValueError, match=r"\$_NAND_ in cell g1"):
        read_netlist(path)
