"""The CUDA kernel against the C reference on the packed words themselves."""

import numpy as np
import pytest
from helpers import DESIGNS, RESET, load

from bitlane import native
from bitlane.levels import pack
from bitlane.stimulus import random_inputs


@pytest.mark.parametrize("name", DESIGNS)
def test_gpu_words_match_c_reference(name):
    netlist, _ = load(name)
    packed = pack(netlist)
    # 1000 tests is 31 full words and one partly used one; 20 cycles exercise the flops.
    inputs = random_inputs(netlist, n_cycles=20, n_tests=1000, reset=RESET.get(name))
    stim_net, stim_words = native.pack_stimulus(packed, inputs)
    probe_net = native.probe_nets(netlist)
    c_words = native.run(packed, stim_net, stim_words, probe_net)
    gpu_words = native.run(packed, stim_net, stim_words, probe_net, gpu=True)
    # On a mismatch this says how many words differ, and where.
    np.testing.assert_array_equal(gpu_words, c_words)
