from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any, cast

from pint import UnitRegistry, get_application_registry, set_application_registry
from pint.compat import np
from pint.facets.plain import PlainQuantity
from pint.testsuite import helpers

pytestmark = (helpers.requires_numpy, helpers.requires_numpy_at_least("2.0"))


@helpers.allprotos
def test_quantity_reduce_numpy_array_roundtrip_matches_pickle_protocol(protocol):
    original_application_registry = get_application_registry()
    ureg = UnitRegistry()
    set_application_registry(ureg)

    try:
        quantity = ureg.Quantity(
            np.arange(6, dtype=np.float64).reshape(2, 3), "meter"
        )

        raw_reduce = quantity.__reduce__()
        assert len(raw_reduce) == 2

        rebuild, rebuild_args = cast(
            tuple[Callable[..., Any], tuple[type[Any], Any, Any]],
            raw_reduce,
        )

        assert rebuild_args[0] is PlainQuantity
        assert isinstance(rebuild_args[1], np.ndarray)

        rebuilt_from_reduce = rebuild(*rebuild_args)
        rebuilt_from_pickle = pickle.loads(pickle.dumps(quantity, protocol))

        helpers.assert_quantity_equal(rebuilt_from_reduce, quantity)
        helpers.assert_quantity_equal(rebuilt_from_pickle, quantity)
        assert rebuilt_from_reduce.magnitude.dtype == quantity.magnitude.dtype
        assert rebuilt_from_pickle.magnitude.dtype == quantity.magnitude.dtype
        assert rebuilt_from_reduce.magnitude.shape == quantity.magnitude.shape
        assert rebuilt_from_pickle.magnitude.shape == quantity.magnitude.shape
    finally:
        set_application_registry(original_application_registry)
