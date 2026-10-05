"""The packed-bit simulator against integer models."""

import numpy as np
import pytest
from helpers import load

from bitlane.levels import pack
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
