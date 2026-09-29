"""Random stimulus: one integer per input port, per cycle, per test."""

import numpy as np

from bitlane.netlist import Netlist


def random_inputs(
    netlist: Netlist,
    n_cycles: int,
    n_tests: int,
    reset: str | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """inputs[port] of shape (n_cycles, n_tests), uniform over the port's width.

    The clock is not driven. The reset port, if named, is held at 1 in cycle 0 so
    that every test starts from reset.
    """
    rng = np.random.default_rng(seed)
    shape = (n_cycles, n_tests)
    inputs = {}
    for name, nets in netlist.inputs.items():
        if name != netlist.clock:
            inputs[name] = rng.integers(0, 2 ** len(nets), shape, dtype=np.uint64)
    if reset is not None:
        if reset not in inputs:
            raise ValueError(f"{reset!r} is not a driven input port")
        inputs[reset][0] = 1
    return inputs
