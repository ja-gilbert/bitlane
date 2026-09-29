"""Random stimulus: port widths, the undriven clock, and reset in cycle 0."""

from helpers import load

from bitlane.stimulus import random_inputs


def test_random_inputs_fit_their_ports():
    counter, _ = load("counter")
    inputs = random_inputs(counter, n_cycles=5, n_tests=100, reset="rst")
    assert set(inputs) == {"rst", "en"}  # the clock is not driven
    assert all(v.shape == (5, 100) and v.max() <= 1 for v in inputs.values())
    assert inputs["rst"][0].min() == 1

    adder, _ = load("adder")
    inputs = random_inputs(adder, n_cycles=1, n_tests=1000)
    assert inputs["a"].max() <= 255 and inputs["cin"].max() <= 1
