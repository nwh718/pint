import sys
import pickle
import pytest
import numpy as np
from typing import Any, Callable, Tuple

try:
    import pint
except ImportError:
    pass

@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Test requires Python 3.14+ for __reduce__ type tightening scenario"
)
def test_quantity_reduce_numpy_2_0_py314_compat():
    """
    Test __reduce__ serialization interface with NumPy 2.0 arrays
    in a Python 3.14 environment.
    
    This test simulates the CI pyright error scenario where type tightening
    in Python 3.14 and NumPy 2.0 might break the __reduce__ serialization
    interface compatibility.
    
    Related PRs/Issues:
    - #2311 (ci.yml: add pyright type checking)
    - #2285 (bench.yml: update GitHub Actions version)
    - #2280 (docs.yml: require NumPy 2.0+)
    """
    ureg = pint.UnitRegistry()
    
    # Create a Quantity with a NumPy 2.0 array
    arr = np.array([1.0, 2.5, 3.2], dtype=np.float64)
    q = ureg.Quantity(arr, "kilogram")
    
    # Simulate pyright error scenario:
    # Under Python 3.14 typeshed updates, __reduce__ return type is strictly enforced.
    # Pyright might report:
    # error: Expression of type "tuple[...]" cannot be assigned to return type "str | tuple[...]"
    # We use explicit type annotations to verify that the runtime behavior
    # still perfectly matches what pickle expects, despite any type hint tightening.
    
    # The return type of __reduce__ is expected to be a tuple with at least 2 elements
    # for a custom object.
    reduce_res: Tuple[Callable[..., Any], Tuple[Any, ...]] = q.__reduce__()  # type: ignore # pyright: ignore[reportAssignmentType]
    
    assert isinstance(reduce_res, tuple), "__reduce__ should return a tuple"
    assert len(reduce_res) >= 2, "__reduce__ tuple should have at least 2 elements"
    
    constructor = reduce_res[0]
    args = reduce_res[1]
    
    assert callable(constructor), "First element must be callable"
    assert isinstance(args, tuple), "Second element must be a tuple of arguments"
    
    # Verify that actual serialization/deserialization works correctly
    # ensuring runtime compatibility is not broken with NumPy 2.0
    serialized = pickle.dumps(q)
    deserialized = pickle.loads(serialized)
    
    # Check if deserialized object is equivalent to the original
    np.testing.assert_array_equal(deserialized.magnitude, q.magnitude)
    assert deserialized.units == q.units
    assert deserialized.magnitude.dtype == q.magnitude.dtype
