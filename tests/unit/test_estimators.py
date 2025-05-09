"""Testing the selfimplemented rotations against scipy rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


@pytest.mark.unit
def test_placeholder():
    """Placeholder test."""
    pass
