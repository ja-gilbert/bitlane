"""The reader on real designs (synthesized at test time) and hand-made edge cases."""

import json
from collections import Counter

import pytest
from helpers import load

from bitlane.netlist import read_netlist
from bitlane.synth import cell_counts


def widths(ports: dict[str, list[int]]) -> dict[str, int]:
    return {name: len(bits) for name, bits in ports.items()}


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


# Hand-made JSON in Yosys's shape for the edge cases.


def port(direction, *bits):
    return {"direction": direction, "bits": list(bits)}


def cell(cell_type, **pins):
    return {"type": cell_type, "connections": {pin: [bit] for pin, bit in pins.items()}}


def read(tmp_path, ports, cells=None, netnames=None):
    module = {"ports": ports, "cells": cells or {}, "netnames": netnames or {}}
    path = tmp_path / "module.json"
    path.write_text(json.dumps({"modules": {"m": module}}))
    return read_netlist(path)


def test_unknown_cell_type(tmp_path):
    with pytest.raises(ValueError, match=r"\$_NAND_ in cell g1"):
        read(tmp_path, {}, {"g1": cell("$_NAND_")})


def test_constant_and_passthrough_outputs(tmp_path):
    # assign y = 1'b1; assign z = 1'b0; assign w = a;
    ports = {
        "a": port("input", 2),
        "y": port("output", "1"),
        "z": port("output", "0"),
        "w": port("output", 2),
    }
    netlist = read(tmp_path, ports)
    assert netlist.outputs == {"y": [1], "z": [0], "w": netlist.inputs["a"]}
    assert netlist.n_nets == 3


def test_x_bit_is_zero_with_a_warning(tmp_path):
    with pytest.warns(UserWarning, match="y is x"):
        netlist = read(tmp_path, {"y": port("output", "x")})
    assert netlist.outputs["y"] == [0]


def test_z_bit_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="y is 'z'"):
        read(tmp_path, {"y": port("output", "z")})


def test_two_clocks_is_an_error(tmp_path):
    ports = {
        "c1": port("input", 2),
        "c2": port("input", 3),
        "d": port("input", 4),
        "q": port("output", 5, 6),
    }
    cells = {
        "f1": cell("$_DFF_P_", C=2, D=4, Q=5),
        "f2": cell("$_DFF_P_", C=3, D=4, Q=6),
    }
    with pytest.raises(ValueError, match="2 different clock nets"):
        read(tmp_path, ports, cells)


def test_undriven_output_is_an_error(tmp_path):
    # A typo like assign {count, sum} = a + b + cin; leaves the cout port undriven.
    ports = {"a": port("input", 2), "cout": port("output", 3)}
    netnames = {"cout": {"hide_name": 0, "bits": [3]}}
    with pytest.raises(ValueError, match="no driver for cout"):
        read(tmp_path, ports, netnames=netnames)
