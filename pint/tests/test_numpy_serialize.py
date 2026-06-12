"""
Test that pint's __reduce__ serialization interface works correctly with NumPy 2.0
arrays in Python 3.14, specifically testing for type narrowing issues that pyright
might flag.

This test simulates the CI scenario introduced by:
- #2311: Add pyright type checking to CI
- #2285: Update GitHub Actions versions in bench.yml
- #2280: Require NumPy 2.0+ in docs.yml

The core issue: PlainQuantity.__reduce__ declares its return type as
``tuple[type, Magnitude, UnitsContainer]`` (a 3-tuple), but the __reduce__
protocol requires returning ``(callable, args_tuple)`` (a 2-tuple).  pyright
flags this structural mismatch.  With NumPy 2.0, the Magnitude type alias
resolves to ``Scalar | np.ndarray[Any, Any]``, and type narrowing of ndarray
generic parameters adds further complexity.
"""

from __future__ import annotations

import pickle
import sys
import typing

import pytest

from pint import Quantity, UnitRegistry
from pint.compat import HAS_NUMPY, np

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="Requires NumPy")


class TestNumpySerialize:
    """Test pickle serialization of Quantity with NumPy array magnitudes."""

    @classmethod
    def setup_class(cls):
        cls.ureg = UnitRegistry()
        cls.Q_ = cls.ureg.Quantity

    def test_pickle_numpy_array_roundtrip(self):
        """Test basic pickle roundtrip with numpy array magnitude."""
        q = self.Q_(np.array([1.0, 2.0, 3.0]), "meter")

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            data = pickle.dumps(q, protocol)
            q2 = pickle.loads(data)

            np.testing.assert_array_equal(q.magnitude, q2.magnitude)
            assert q.units == q2.units

    def test_pickle_numpy_2d_array(self):
        """Test pickling with 2D numpy arrays."""
        q = self.Q_(np.array([[1.0, 2.0], [3.0, 4.0]]), "kg")

        data = pickle.dumps(q)
        q2 = pickle.loads(data)

        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units

    def test_pickle_numpy_integer_array(self):
        """Test pickling with integer numpy arrays."""
        q = self.Q_(np.array([1, 2, 3], dtype=np.int64), "second")

        data = pickle.dumps(q)
        q2 = pickle.loads(data)

        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units
        assert q2.magnitude.dtype == np.int64

    def test_pickle_preserves_numpy_dtype(self):
        """Test that numpy dtype is preserved through pickling.

        NumPy 2.0 introduces stricter dtype typing.  This test ensures
        that dtype narrowing does not affect serialization correctness.
        """
        for dtype in [np.float64, np.float32, np.int32, np.int64, np.complex128]:
            q = self.Q_(np.array([1, 2, 3], dtype=dtype), "meter")
            data = pickle.dumps(q)
            q2 = pickle.loads(data)
            assert q2.magnitude.dtype == dtype, (
                f"dtype {dtype} was not preserved through pickling. "
                f"Got {q2.magnitude.dtype} instead."
            )

    def test_pickle_zero_dim_numpy_array(self):
        """Test pickling with 0-dimensional numpy arrays.

        NumPy 2.0 has stricter handling of 0-d array types.  This test
        ensures serialization remains compatible.
        """
        q = self.Q_(np.array(5.0), "meter")
        data = pickle.dumps(q)
        q2 = pickle.loads(data)
        assert q2.magnitude == 5.0
        assert q.units == q2.units

    def test_pickle_byteswapped_numpy_array(self):
        """Test pickling with non-native byte order arrays.

        NumPy 2.0 type narrowing may affect arrays with non-native byte order.
        """
        arr = np.array([1.0, 2.0, 3.0], dtype=">f8")
        q = self.Q_(arr, "meter")
        data = pickle.dumps(q)
        q2 = pickle.loads(data)
        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units

    def test_pickle_structured_numpy_array(self):
        """Test pickling with structured numpy arrays.

        NumPy 2.0 has improved type annotations for structured dtypes.
        """
        dtype = np.dtype([("x", np.float64), ("y", np.int32)])
        arr = np.array([(1.0, 2), (3.0, 4)], dtype=dtype)
        q = self.Q_(arr, "meter")
        data = pickle.dumps(q)
        q2 = pickle.loads(data)
        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units

    def test_reduce_return_structure(self):
        """Test that __reduce__ returns correct structure for pickling protocol.

        The __reduce__ protocol requires a 2-tuple (callable, args_tuple).
        However, PlainQuantity.__reduce__ is annotated as returning a 3-tuple
        ``tuple[type, Magnitude, UnitsContainer]``.  pyright (#2311) flags
        this structural mismatch.

        This test verifies the runtime behaviour is correct regardless of the
        annotation error.
        """
        q = self.Q_(np.array([1.0, 2.0, 3.0]), "meter")

        result = q.__reduce__()

        assert isinstance(result, tuple)
        assert len(result) == 2, (
            "__reduce__ must return a 2-tuple (callable, args_tuple), "
            f"got {len(result)}-tuple.  This is the pyright type error "
            "scenario: the annotation says tuple[type, Magnitude, "
            "UnitsContainer] (3-tuple) but the actual return is "
            "(callable, (cls, magnitude, units)) (2-tuple)."
        )

        callable_obj, args = result
        assert callable(callable_obj), "First element must be a callable"
        assert isinstance(args, tuple), "Second element must be a tuple of args"

        assert len(args) == 3, (
            f"args tuple must have 3 elements (cls, magnitude, units), "
            f"got {len(args)}"
        )

        cls, magnitude, units = args

        assert isinstance(magnitude, np.ndarray), (
            "Magnitude must be a numpy array.  "
            "NumPy 2.0 type narrowing may cause pyright to flag this."
        )

        reconstructed = callable_obj(*args)
        assert isinstance(reconstructed, self.Q_)
        np.testing.assert_array_equal(q.magnitude, reconstructed.magnitude)
        assert q.units == reconstructed.units

    def test_reduce_type_annotation_mismatch(self):
        """Simulate pyright type checking of __reduce__ return type.

        Uses ``typing.get_type_hints`` to inspect the declared return type
        of ``__reduce__`` and compares it with the actual runtime structure.
        This replicates what pyright does statically in CI.

        The annotation ``tuple[type, Magnitude, UnitsContainer]`` implies a
        3-tuple, but __reduce__ actually returns a 2-tuple.  pyright would
        report: "Tuple size mismatch: expected 3, got 2".
        """
        q = self.Q_(np.array([1.0, 2.0, 3.0]), "meter")

        hints = typing.get_type_hints(Quantity.__reduce__)
        annotated_return = hints.get("return")

        result = q.__reduce__()

        if annotated_return is not None:
            origin = typing.get_origin(annotated_return)
            if origin is tuple:
                annotated_len = len(typing.get_args(annotated_return))
                actual_len = len(result)

                assert annotated_len != actual_len, (
                    f"Expected annotation mismatch: __reduce__ return type "
                    f"annotation says {annotated_len}-tuple but actual return "
                    f"is {actual_len}-tuple.  If this assertion fails, the "
                    f"annotation has been fixed (good!) and this test should "
                    f"be updated."
                )

        callable_obj, args = result
        assert callable(callable_obj)
        assert isinstance(args[1], np.ndarray)

        reconstructed = callable_obj(*args)
        np.testing.assert_array_equal(q.magnitude, reconstructed.magnitude)
        assert q.units == reconstructed.units

    @pytest.mark.skipif(
        sys.version_info < (3, 14),
        reason="Python 3.14+ required for stricter type narrowing behaviour",
    )
    def test_python314_type_narrowing(self):
        """Test serialization under Python 3.14's stricter type narrowing.

        Python 3.14 may introduce stricter type checking at runtime or
        through updated type stubs.  This test ensures the serialization
        interface remains compatible with NumPy 2.0 arrays.
        """
        q = self.Q_(np.array([1.0, 2.0, 3.0], dtype=np.float64), "meter")

        data = pickle.dumps(q)
        q2 = pickle.loads(data)

        assert q2.magnitude.dtype == np.float64
        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units

    def test_reduce_magnitude_type_compatibility(self):
        """Test that the magnitude type in __reduce__ args is compatible
        with the Magnitude type alias used in the annotation.

        In NumPy 2.0, ``np.ndarray`` is generic over shape and dtype.
        ``self.magnitude`` may be ``np.ndarray[tuple[int], np.dtype[np.float64]]``
        which is a subtype of ``np.ndarray[Any, Any]`` used in ``Magnitude``.
        pyright should accept this, but type narrowing could cause issues.
        """
        for dtype in [np.float64, np.int64, np.float32, np.complex128]:
            q = self.Q_(np.array([1.0, 2.0, 3.0], dtype=dtype), "meter")

            _callable, args = q.__reduce__()
            _cls, magnitude, _units = args

            assert isinstance(magnitude, np.ndarray), (
                f"Magnitude with dtype {dtype} must be an np.ndarray instance. "
                f"NumPy 2.0 type narrowing: got {type(magnitude)}."
            )
            assert magnitude.dtype == dtype, (
                f"dtype mismatch after __reduce__: expected {dtype}, "
                f"got {magnitude.dtype}."
            )

    def test_pickle_masked_array(self):
        """Test pickling with numpy masked arrays.

        Masked arrays are a NumPy subclass with additional type complexity
        that may trigger pyright type narrowing issues.
        """
        arr = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        q = self.Q_(arr, "meter")
        data = pickle.dumps(q)
        q2 = pickle.loads(data)

        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        np.testing.assert_array_equal(q.magnitude.mask, q2.magnitude.mask)
        assert q.units == q2.units

    def test_pickle_record_array(self):
        """Test pickling with numpy record arrays.

        Record arrays are structured arrays with attribute access, which
        NumPy 2.0 types more strictly.
        """
        arr = np.rec.array([(1.0, 2), (3.0, 4)], dtype=[("x", float), ("y", int)])
        q = self.Q_(arr, "meter")
        data = pickle.dumps(q)
        q2 = pickle.loads(data)

        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units

    def test_pickle_fortran_contiguous_array(self):
        """Test pickling with Fortran-contiguous (column-major) arrays.

        NumPy 2.0's stricter type annotations account for memory layout
        through flags, potentially causing type narrowing issues.
        """
        arr = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert arr.flags["F_CONTIGUOUS"]
        q = self.Q_(arr, "meter")

        data = pickle.dumps(q)
        q2 = pickle.loads(data)

        np.testing.assert_array_equal(q.magnitude, q2.magnitude)
        assert q.units == q2.units