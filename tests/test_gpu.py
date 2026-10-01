"""The CUDA kernel against the C reference on the packed words, and warm runs."""

import numpy as np
import pytest
from helpers import DESIGNS, RESET, load

from bitlane import native
from bitlane.levels import pack
from bitlane.stimulus import random_inputs

BACKENDS = pytest.mark.parametrize("gpu", [False, True], ids=["c", "gpu"])


def open_design(name, gpu, seed=0):
    """A session for `name` and its packed stimulus: 1000 tests (31 full words and a
    partial one) over 20 cycles."""
    netlist, _ = load(name)
    packed = pack(netlist)
    inputs = random_inputs(netlist, 20, 1000, reset=RESET.get(name), seed=seed)
    stim_net, stim_words = native.pack_stimulus(packed, inputs)
    n_cycles, _, n_words = stim_words.shape
    probe_net = native.probe_nets(netlist)
    session = native.open_session(packed, stim_net, probe_net, n_cycles, n_words, gpu)
    return session, stim_words


@pytest.mark.parametrize("name", DESIGNS)
def test_gpu_words_match_c_reference(name):
    words = {}
    for gpu in (False, True):
        session, stim_words = open_design(name, gpu)
        words[gpu] = native.run(session, stim_words).copy()
        native.close_session(session)
    # On a mismatch this says how many words differ, and where.
    np.testing.assert_array_equal(words[True], words[False])


@BACKENDS
@pytest.mark.parametrize("name", ["counter", "fsm"])  # the designs with state
def test_warm_runs_start_from_reset(name, gpu):
    session, stim_words = open_design(name, gpu)
    other_session, other_words = open_design(name, gpu, seed=1)
    native.close_session(other_session)
    first = native.run(session, stim_words).copy()
    native.run(session, other_words)  # leaves different state in the flops
    again = native.run(session, stim_words).copy()
    native.close_session(session)
    np.testing.assert_array_equal(again, first)


@BACKENDS
def test_run_refuses_a_wrong_shape_or_a_closed_session(gpu):
    session, stim_words = open_design("counter", gpu)
    with pytest.raises(ValueError, match="stimulus"):
        # Half the cycles: without the check the C code would read past the array.
        native.run(session, stim_words[:10].copy())
    native.close_session(session)
    native.close_session(session)  # closing twice is harmless
    with pytest.raises(ValueError, match="closed"):
        native.run(session, stim_words)
