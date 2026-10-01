"""Every test that takes `simulate` runs once per simulator: NumPy, C, GPU."""

from functools import partial

import pytest

from bitlane import native, refsim


@pytest.fixture(
    params=[refsim.simulate, native.simulate, partial(native.simulate, gpu=True)],
    ids=["numpy", "c", "gpu"],
)
def simulate(request):
    return request.param
