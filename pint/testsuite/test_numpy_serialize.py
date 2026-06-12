"""Test __reduce__ serialization compatibility with NumPy 2.0 arrays.

This test validates that pint's __reduce__ serialization interface does not
break compatibility with NumPy 2.0 arrays when type annotations are tightened
(as enforced by pyright in CI #2311).

Related issues:
- #2311: pyright type checking
- #2285: GitHub Actions version updates
- #2280: NumPy 2.0+ requirement
"""

from __future__ import annotations

import pickle
import sys
from typing import get_args, get_origin

import pytest

from pint import UnitRegistry, get_application_registry, set_application_registry
from pint.compat import HAS_NUMPY, np
from pint.facets.plain.quantity import PlainQuantity
from pint.util import UnitsContainer


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
class TestNumpySerializeReduce:
    """Test __reduce__ serialization with NumPy arrays."""

    @classmethod
    def setup_class(cls):
        cls.ureg = UnitRegistry()

    def test_reduce_return_type_structure(self):
        """Validate __reduce__ returns correct tuple structure for pickle protocol."""
        arr = np.array([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "meter")

        result = q.__reduce__()

        assert isinstance(result, tuple)
        assert len(result) == 2

        func, args = result
        assert callable(func)
        assert isinstance(args, tuple)
        assert len(args) == 3

        cls_arg, magnitude_arg, units_arg = args
        assert cls_arg is PlainQuantity
        assert isinstance(magnitude_arg, np.ndarray)
        assert isinstance(units_arg, UnitsContainer)

    def test_reduce_type_annotation_compatibility(self):
        """Verify __reduce__ return type matches pyright expectations.

        Pyright (#2311) enforces strict type checking. The return type annotation
        should be compatible with Python's pickle protocol requirements.
        """
        import inspect

        sig = inspect.signature(PlainQuantity.__reduce__)
        return_annotation = sig.return_annotation

        if return_annotation is not inspect.Signature.empty:
            origin = get_origin(return_annotation)
            if origin is tuple:
                args = get_args(return_annotation)
                assert len(args) == 3, (
                    f"Expected 3 type args in __reduce__ return, got {len(args)}"
                )

    def test_pickle_roundtrip_numpy_array(self):
        """Test pickle roundtrip preserves NumPy 2.0 array data and dtype."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        q = self.ureg.Quantity(arr, "meter")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        assert isinstance(restored, self.ureg.Quantity)
        np.testing.assert_array_equal(restored.magnitude, arr)
        assert restored.magnitude.dtype == arr.dtype
        assert restored.units == q.units

    def test_pickle_roundtrip_preserves_array_type(self):
        """Ensure unpickling returns numpy.ndarray, not generic sequence."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        q = self.ureg.Quantity(arr, "second")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        assert isinstance(restored.magnitude, np.ndarray)
        assert restored.magnitude.dtype == np.int32
        assert restored.magnitude.shape == (2, 2)

    def test_pickle_roundtrip_with_application_registry(self):
        """Test that unpickling uses application registry correctly."""
        original_registry = get_application_registry()

        try:
            set_application_registry(self.ureg)

            arr = np.array([10.0, 20.0])
            q = self.ureg.Quantity(arr, "kilometer")

            serialized = pickle.dumps(q)
            restored = pickle.loads(serialized)

            assert isinstance(restored, self.ureg.Quantity)
            np.testing.assert_array_equal(restored.magnitude, arr)
            assert str(restored.units) == "kilometer"
        finally:
            set_application_registry(original_registry)

    def test_pickle_various_numpy_dtypes(self):
        """Test serialization compatibility across NumPy 2.0 dtype spectrum."""
        dtypes_to_test = [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.complex128,
            np.bool_,
        ]

        for dtype in dtypes_to_test:
            arr = np.array([1, 2, 3], dtype=dtype)
            q = self.ureg.Quantity(arr, "meter")

            serialized = pickle.dumps(q)
            restored = pickle.loads(serialized)

            assert isinstance(restored.magnitude, np.ndarray)
            assert restored.magnitude.dtype == dtype
            np.testing.assert_array_equal(
                restored.magnitude, arr,
                err_msg=f"Failed for dtype {dtype}"
            )

    def test_pickle_zero_dimensional_array(self):
        """Test serialization of 0-dimensional numpy arrays."""
        arr = np.array(42.0)
        q = self.ureg.Quantity(arr, "joule")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        assert isinstance(restored.magnitude, np.ndarray)
        assert restored.magnitude.ndim == 0
        assert restored.magnitude.item() == 42.0

    def test_pickle_empty_array(self):
        """Test serialization of empty numpy arrays."""
        arr = np.array([])
        q = self.ureg.Quantity(arr, "meter")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        assert isinstance(restored.magnitude, np.ndarray)
        assert restored.magnitude.size == 0

    def test_reduce_magnitude_is_array_not_quantity(self):
        """Verify __reduce__ magnitude arg is raw array, not wrapped."""
        arr = np.array([1.0, 2.0])
        q = self.ureg.Quantity(arr, "meter")

        func, args = q.__reduce__()
        _, magnitude, _ = args

        assert isinstance(magnitude, np.ndarray)
        assert not hasattr(magnitude, "_units")
        assert not hasattr(magnitude, "magnitude")

    def test_pickle_protocol_versions(self):
        """Test compatibility across pickle protocol versions."""
        arr = np.array([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "meter")

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            serialized = pickle.dumps(q, protocol=protocol)
            restored = pickle.loads(serialized)

            np.testing.assert_array_equal(restored.magnitude, arr)
            assert restored.units == q.units


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
class TestNumpy2Compatibility:
    """Test NumPy 2.0+ specific compatibility scenarios."""

    @classmethod
    def setup_class(cls):
        cls.ureg = UnitRegistry()

    def test_numpy_version_check(self):
        """Verify we're testing against NumPy 2.0+ as required by #2280."""
        version_parts = np.__version__.split(".")
        major = int(version_parts[0])
        assert major >= 2, f"Expected NumPy 2.0+, got {np.__version__}"

    def test_array_api_compatibility(self):
        """Test that serialized arrays maintain array API compliance."""
        arr = np.array([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "meter")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        assert hasattr(restored.magnitude, "__array__")
        assert hasattr(restored.magnitude, "dtype")
        assert hasattr(restored.magnitude, "shape")

    def test_strided_array_serialization(self):
        """Test serialization of strided numpy arrays."""
        original = np.array([0, 1, 2, 3, 4, 5])
        strided = original[::2]

        q = self.ureg.Quantity(strided, "meter")
        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        np.testing.assert_array_equal(restored.magnitude, [0, 2, 4])

    def test_fortran_order_array_serialization(self):
        """Test serialization of Fortran-order arrays."""
        arr = np.array([[1, 2], [3, 4]], order="F")
        q = self.ureg.Quantity(arr, "meter")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        np.testing.assert_array_equal(restored.magnitude, arr)


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Python 3.14+ required for strict type checking simulation"
)
class TestPython314StrictTypes:
    """Test scenarios specific to Python 3.14 strict type checking."""

    @classmethod
    def setup_class(cls):
        cls.ureg = UnitRegistry()

    def test_reduce_return_type_runtime_check(self):
        """Runtime validation of __reduce__ return type under Python 3.14."""
        arr = np.array([1.0, 2.0])
        q = self.ureg.Quantity(arr, "meter")

        result = q.__reduce__()

        assert isinstance(result, tuple)

        func, args = result
        assert callable(func)
        assert isinstance(args, tuple)
        assert len(args) == 3

        cls_arg, magnitude, units = args
        assert cls_arg is PlainQuantity
        assert isinstance(magnitude, np.ndarray)
        assert isinstance(units, UnitsContainer)

    def test_pickle_with_strict_type_annotations(self):
        """Verify pickle works with strict type annotation enforcement."""
        arr = np.array([1.5, 2.5, 3.5])
        q = self.ureg.Quantity(arr, "meter")

        serialized = pickle.dumps(q)
        restored = pickle.loads(serialized)

        assert type(restored) is type(q)
        np.testing.assert_array_equal(restored.magnitude, arr)
