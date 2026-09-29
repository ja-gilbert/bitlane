"""Every test that takes `simulate` runs once per simulator."""

import pytest

from bitlane import native, refsim


@pytest.fixture(params=[refsim.simulate, native.simulate], ids=["numpy", "c"])
def simulate(request):
    return request.param
