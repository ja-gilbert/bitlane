"""The packed-bit simulator against integer models."""

import numpy as np
import pytest
from helpers import load

from bitlane.levels import pack
from bitlane.netlist import Gate, Netlist
from bitlane.refsim import evaluate


@pytest.mark.parametrize("n_tests", [4096, 100])  # 100 is not a multiple of 32
def test_adder_matches_integer_addition(n_tests):
    netlist, _ = load("adder")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, n_tests, dtype=np.uint64)
    b = rng.integers(0, 256, n_tests, dtype=np.uint64)
    cin = rng.integers(0, 2, n_tests, dtype=np.uint64)
    out = evaluate(pack(netlist), {"a": a, "b": b, "cin": cin})
    total = a + b + cin
    assert np.array_equal(out["sum"], total & 0xFF)
    assert np.array_equal(out["cout"], total >> 8)


def test_mux_and_not():
    # The smallest netlist with a MUX and a NOT: y = s ? b : a and z = NOT a
    # Nets: 2 a, 3 b, 4 s, 5 y, 6 z.
    netlist = Netlist(
        n_nets=7,
        inputs={"a": [2], "b": [3], "s": [4]},
        outputs={"y": [5], "z": [6]},
        clock=None,
        gates=[Gate("MUX", [2, 3, 4], 5), Gate("NOT", [2], 6)],
        flops=[],
    )
    rng = np.random.default_rng(1)
    a, b, s = rng.integers(0, 2, (3, 50), dtype=np.uint64)
    out = evaluate(pack(netlist), {"a": a, "b": b, "s": s})
    assert np.array_equal(out["y"], np.where(s == 1, b, a))
    assert np.array_equal(out["z"], 1 - a)


def test_counter_matches_integer_model(simulate):
    netlist, _ = load("counter")
    rng = np.random.default_rng(2)
    n_cycles, n_tests = 600, 1024  # long enough for the count to wrap past 255
    rst = (rng.random((n_cycles, n_tests)) < 0.003).astype(np.uint64)
    en = (rng.random((n_cycles, n_tests)) < 0.9).astype(np.uint64)
    rst[0] = 1  # reset first, like the Icarus testbench will
    out = simulate(pack(netlist), {"rst": rst, "en": en})
    count = np.zeros(n_tests, dtype=np.uint64)
    for cycle in range(n_cycles):
        # Same order as the simulator: this cycle's output is sampled before the edge.
        assert np.array_equal(out["count"][cycle], count)
        count = np.where(rst[cycle] == 1, 0, (count + en[cycle]) & 0xFF)
    counts = out["count"]  # and a real 255 -> 0 step without a reset happened
    assert ((counts[:-1] == 255) & (counts[1:] == 0) & (rst[:-1] == 0)).any()
