"""pint.pint.tests.test_numpy_serialize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression / compatibility tests covering the interaction between:

* ``PlainQuantity.__reduce__`` (``pint/facets/plain/quantity.py``)
* The ``Magnitude`` type alias (``pint/_typing.py``)
* NumPy 2.x ``ndarray`` magnitudes
* Python 3.14's stricter pickle / attribute bookkeeping

The CI ``ci.yml`` now runs ``pyright`` for type checking (see the
``typecheck`` job).  ``__reduce__`` is currently annotated as returning
``tuple[type, Magnitude, UnitsContainer]`` (a 3-tuple ``(cls, mag, units)``
which would correspond to pickle protocol form 0), but the actual
implementation returns a **2-tuple** ``(callable, args)`` per pickle
protocol form 2, and the ``args`` payload itself is a 3-tuple containing
the magnitude.  When the magnitude is a ``numpy.ndarray`` this produces
two observable problems under pyright / NumPy 2.0:

1. The declared return type does not match the actual return type —
   pyright emits an error at the typecheck stage.
2. The ``Magnitude`` type alias is ``Scalar | Array``; inside the args
   tuple the array is typed ``Array`` (``np.ndarray[Any, Any]``), but
   ``UnitsContainer`` is ``dict[str, Scalar]``-like.  When ``__reduce__``
   places an ndarray next to a ``UnitsContainer`` inside a plain tuple,
   pyright on NumPy 2.0 no longer widens ``ndarray`` to ``Any`` the way
   it did on NumPy 1.x, so tuple-unification of ``(PlainQuantity,
   ndarray, UnitsContainer)`` against ``tuple[type, Magnitude,
   UnitsContainer]`` produces a diagnostic.
3. At runtime on Python 3.14 + NumPy 2.0 the pickle machinery is stricter
   about dunder protocols (see CPython issue #118019 / NumPy 2.0 release
   notes regarding ``__reduce_ex__`` dispatch); we therefore exercise
   *all* pickle protocols end-to-end with ndarray-backed quantities.

This module contains pytest cases that reproduce those failure modes and
guard against future regressions.
"""

from __future__ import annotations

import copy
import pickle
import sys
import types
from typing import Any, get_args, get_origin

import pytest

import pint
from pint import Quantity, UnitRegistry
from pint._typing import Magnitude
from pint.compat import HAS_NUMPY, np
from pint.facets.plain.quantity import PlainQuantity
from pint.util import UnitsContainer


requires_numpy = pytest.mark.skipif(
    not HAS_NUMPY, reason="Requires NumPy (test targets NumPy 2.0+)"
)


requires_numpy2 = pytest.mark.skipif(
    not (HAS_NUMPY and np.__version__.split(".")[0] >= "2"),
    reason="Requires NumPy >= 2.0",
)


requires_py314 = pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Only validates the Python 3.14 pickle behaviour",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_pickle_protocol2_form(reduced: Any) -> bool:
    """Return True when ``reduced`` looks like pickle protocol form 2.

    ``(callable, args)`` is the expected shape for ``__reduce__`` form 2
    (Python docs / :pep:`307`).  callable must be callable and args must
    be a tuple.
    """
    return (
        isinstance(reduced, tuple)
        and len(reduced) == 2
        and callable(reduced[0])
        and isinstance(reduced[1], tuple)
    )


def _is_pickle_protocol0_form(reduced: Any) -> bool:
    """Return True when ``reduced`` looks like pickle protocol form 0.

    ``(cls, (mag, units))`` is the classical form 0/1 shape where the
    class is instantiated with two positional arguments.
    """
    return (
        isinstance(reduced, tuple)
        and len(reduced) == 2
        and isinstance(reduced[0], type)
        and isinstance(reduced[1], tuple)
    )


def _is_annotation_shape(reduced: Any) -> bool:
    """True if ``reduced`` matches the declared ``tuple[type, Magnitude, UnitsContainer]``.

    That is a *flat* 3-tuple ``(type, magnitude_value, units_container)``
    — NOT wrapped in a second args tuple.  This is what the current type
    annotation promises but what the implementation does not return.
    """
    return (
        isinstance(reduced, tuple)
        and len(reduced) == 3
        and isinstance(reduced[0], type)
        and isinstance(reduced[2], UnitsContainer)
    )


# ---------------------------------------------------------------------------
# 1. Type-annotation shape vs. runtime shape
# ---------------------------------------------------------------------------


@requires_numpy
class TestReduceShape:
    """Describe the shape of ``PlainQuantity.__reduce__``.

    The tests in this class assert the *actual* runtime behaviour and the
    *declared* type annotation.  They are written to fail if the
    annotation and the implementation disagree — which is exactly what
    the CI ``pyright`` job is catching for NumPy 2.0 magnitudes.
    """

    def setup_method(self):
        self.ureg = UnitRegistry()

    def test_scalar_magnitude_reduce_is_form2(self):
        q = self.ureg.Quantity(42.0, "m")
        reduced = q.__reduce__()

        # Runtime shape is (callable, args) — pickle protocol form 2.
        assert _is_pickle_protocol2_form(reduced), (
            "PlainQuantity.__reduce__ must return a (callable, args) 2-tuple "
            f"for pickle protocol form 2, got: {type(reduced).__mro__} / {reduced!r}"
        )

    def test_ndarray_magnitude_reduce_is_form2(self):
        arr = np.asarray([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "m")
        reduced = q.__reduce__()

        assert _is_pickle_protocol2_form(reduced), (
            "PlainQuantity.__reduce__ with a NumPy ndarray magnitude must "
            "still return a (callable, args) 2-tuple."
        )

        # The magnitude in the args payload must still be the *same*
        # ndarray so that pickle round-trips work end-to-end.
        _callable, args = reduced
        stored_cls, stored_magnitude, stored_units = args
        assert stored_cls is PlainQuantity
        # Magnitude type must be np.ndarray under NumPy 2.0 — not widened
        # to Any, which is what pyright used to accept pre-2.0.
        assert isinstance(stored_magnitude, np.ndarray), (
            "The magnitude stored inside __reduce__ must remain a "
            f"numpy.ndarray, got {type(stored_magnitude).__mro__}"
        )
        assert isinstance(stored_units, UnitsContainer)

    def test_reduce_does_not_match_annotation_shape(self):
        """Demonstrate the type-annotation bug that pyright flags.

        The annotation on ``__reduce__`` claims
        ``tuple[type, Magnitude, UnitsContainer]`` (a *flat* 3-tuple),
        but the runtime produces a *nested* 2-tuple.  We assert this
        mismatch so the guard remains visible until a maintainer decides
        whether to fix the annotation or change the implementation.
        """
        q = self.ureg.Quantity(np.asarray([1.0, 2.0]), "kg")
        reduced = q.__reduce__()

        # This is the assertion that mirrors the pyright diagnostic: the
        # declared shape does NOT hold — if someone ever "fixes" the
        # implementation to match the annotation, this test will flip
        # from FAIL to PASS and the second assertion (the real guard)
        # will start failing, alerting us that pickle round-tripping has
        # regressed.
        assert not _is_annotation_shape(
            reduced
        ), (
            "PlainQuantity.__reduce__ returned a flat (cls, mag, units) "
            "3-tuple — if this was intentional, remove the "
            "test_reduce_does_not_match_annotation_shape guard and make "
            "sure pickle round-trips still succeed below."
        )

    def test_magnitude_type_alias_includes_ndarray(self):
        """Ensure the ``Magnitude`` type alias really does cover ndarray.

        ``Magnitude = Scalar | Array`` where ``Array = np.ndarray[Any,
        Any]`` on NumPy >= 2.0.  If this ever becomes ``Scalar`` only,
        then even a correctly shaped ``__reduce__`` payload would fail
        pyright's type inference for NumPy magnitudes.
        """
        # ``Magnitude`` is defined with the PEP 695 ``type`` keyword on
        # Python 3.12+, so ``get_origin`` / ``get_args`` return
        # ``None`` / ``()`` on it directly — we must go through the
        # alias's ``__value__`` attribute instead, which exposes the
        # underlying ``Scalar | Array`` union.
        effective_type = _resolve_type_alias(Magnitude)
        origin = get_origin(effective_type)
        args = get_args(effective_type)
        assert origin is not None and args, (
            "Magnitude must resolve to a parameterised union at runtime; "
            f"got origin={origin!r}, args={args!r}"
        )
        # At least one arm of the union must name ``np.ndarray`` in
        # some form (bare ``np.ndarray`` or a generic
        # ``np.ndarray[Any, Any]``).
        has_array_arg = any(_candidate_is_array_like(arg) for arg in args)
        assert has_array_arg, (
            "Magnitude type alias must include an ndarray variant so "
            "that __reduce__ payloads with NumPy magnitudes are well-"
            f"typed; got args={args!r}"
        )


# ---------------------------------------------------------------------------
# 2. Runtime pickle round-trip with NumPy 2.0 magnitudes
# ---------------------------------------------------------------------------


@requires_numpy2
class TestNumpy2PickleRoundtrip:
    """Exercise pickle end-to-end with NumPy 2.0 magnitudes.

    These tests are the runtime counterpart to the pyright diagnostics
    above: even if the annotation is wrong, we must still be able to
    round-trip a ``Quantity`` whose magnitude is an ``np.ndarray``
    through :mod:`pickle`, :func:`copy.copy` and :func:`copy.deepcopy`
    under Python 3.14.
    """

    def setup_method(self):
        self.ureg = UnitRegistry()

    @pytest.mark.parametrize(
        "protocol",
        list(range(pickle.HIGHEST_PROTOCOL + 1)),
    )
    def test_ndarray_quantity_pickle_roundtrip_all_protocols(self, protocol):
        arr = np.asarray([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        q = self.ureg.Quantity(arr, "m/s")
        blob = pickle.dumps(q, protocol=protocol)
        restored = pickle.loads(blob)

        assert isinstance(restored, Quantity)
        assert isinstance(restored.magnitude, np.ndarray)
        np.testing.assert_array_equal(restored.magnitude, arr)
        assert restored.units == q.units

    def test_ndarray_quantity_copy_shallow(self):
        arr = np.asarray([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "kg")
        q2 = copy.copy(q)

        np.testing.assert_array_equal(q2.magnitude, arr)
        assert q2.units == q.units
        # copy.copy on a Quantity should share the ndarray (shallow).
        assert q2.magnitude is arr or q2.magnitude.base is arr

    def test_ndarray_quantity_copy_deep(self):
        arr = np.asarray([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "kg")
        q2 = copy.deepcopy(q)

        np.testing.assert_array_equal(q2.magnitude, arr)
        assert q2.units == q.units
        # deepcopy must produce a truly independent ndarray.
        assert q2.magnitude is not arr

    def test_structured_ndarray_quantity_pickle(self):
        """NumPy 2.0 tightened structured-dtype handling; keep it honest."""
        dtype = np.dtype([("x", np.float64), ("y", np.float64)])
        arr = np.asarray([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)
        q = self.ureg.Quantity(arr, "meter")

        restored = pickle.loads(pickle.dumps(q))
        assert isinstance(restored.magnitude, np.ndarray)
        assert restored.magnitude.dtype == dtype
        np.testing.assert_array_equal(restored.magnitude, arr)
        assert restored.units == q.units

    def test_ndarray_quantity_reduce_roundtrip_via_callable(self):
        """Directly call ``_unpickle_quantity`` with the __reduce__ args.

        This bypasses :mod:`pickle` to exercise the exact payload that
        ``__reduce__`` produces, which is what pyright analyses statically.
        """
        q = self.ureg.Quantity(np.asarray([7.0, 8.0, 9.0]), "second")
        callable_, args = q.__reduce__()
        assert callable_ is pint._unpickle_quantity
        restored = callable_(*args)
        np.testing.assert_array_equal(restored.magnitude, q.magnitude)
        assert restored.units == q.units

    def test_ndarray_quantity_reduce_ex_dispatch(self):
        """Python 3.14 is stricter about ``__reduce_ex__`` dispatch.

        Make sure ``Quantity.__reduce_ex__(protocol)`` still produces a
        payload pickle accepts for any supported protocol, even when the
        magnitude is a NumPy 2.0 ndarray.
        """
        q = self.ureg.Quantity(np.asarray([1.0, 2.0, 3.0]), "m")
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            # ``__reduce_ex__`` may return a 2, 3, 4 or 5-tuple depending
            # on the protocol; the first two elements must still be
            # ``(callable, args)`` per the standard.
            reduced = q.__reduce_ex__(protocol)
            assert isinstance(reduced, tuple) and len(reduced) >= 2
            assert callable(reduced[0])
            assert isinstance(reduced[1], tuple)
            # Full round-trip must still succeed.
            restored = pickle.loads(pickle.dumps(q, protocol=protocol))
            np.testing.assert_array_equal(restored.magnitude, q.magnitude)


# ---------------------------------------------------------------------------
# 3. Pyright-style static-consistency guard (runs on all Pythons)
# ---------------------------------------------------------------------------


@requires_numpy
class TestPyrightConsistency:
    """Guards that mirror what the CI ``pyright`` job diagnoses.

    We cannot actually run pyright from inside pytest (we don't want to
    depend on it at test-time), but we can encode the *structural*
    invariants that pyright would check statically as runtime assertions
    on the ``__reduce__`` output and on the ``Magnitude`` alias.  This
    way, if a contributor changes ``__reduce__`` in a way that widens
    the magnitude to ``Any`` or flattens the payload to a 3-tuple, the
    guard here fails before the CI typecheck job does.
    """

    def setup_method(self):
        self.ureg = UnitRegistry()

    def test_reduce_payload_args_tuple_is_three_elements(self):
        """The args nested tuple must always be (cls, magnitude, units)."""
        q = self.ureg.Quantity(np.asarray([1.0, 2.0]), "m")
        _callable, args = q.__reduce__()
        assert len(args) == 3, (
            "Payload args must be a 3-tuple (cls, magnitude, units); "
            f"got {len(args)} elements."
        )
        cls, magnitude, units = args
        assert cls is PlainQuantity
        assert isinstance(magnitude, np.ndarray)
        assert isinstance(units, UnitsContainer)

    def test_reduce_callable_is_registered_module_attribute(self):
        """The callable must live in ``pint`` so pickle can find it by
        name across interpreter instances (otherwise:
        ``AttributeError`` on unpickle in a fresh process).
        """
        q = self.ureg.Quantity(np.asarray([1.0, 2.0]), "m")
        callable_, _args = q.__reduce__()
        assert hasattr(pint, callable_.__name__), (
            f"__reduce__ callable {callable_!r} must be a top-level "
            "attribute of the ``pint`` module so pickle can locate it"
        )
        assert getattr(pint, callable_.__name__) is callable_

    def test_magnitude_alias_not_reduced_to_scalar(self):
        """Regression guard against the accidental removal of the
        ``Array`` arm of ``Magnitude``.  Without that arm, pyright
        complains about every ``Quantity[ndarray]``.
        """
        from pint import _typing

        # The module-level ``Magnitude`` value must resolve to a union
        # with at least one ndarray-shaped arm.  We share the
        # ``_resolve_type_alias`` helper with ``TestReduceShape`` so that
        # PEP 695 ``type`` aliases are correctly unwrapped before the
        # union inspection.
        effective = _resolve_type_alias(_typing.Magnitude)
        origin = get_origin(effective)
        args = get_args(effective)
        if origin is not None and args:
            assert any(
                _candidate_is_array_like(arg) for arg in args
            ), (
                "Magnitude type alias must include an ndarray arm for "
                f"pyright compatibility with NumPy 2.0 magnitudes; got "
                f"args={args!r}"
            )
        # Otherwise ``Magnitude`` has been eagerly resolved (very old
        # Python); the static assertion in ``TestReduceShape`` covers
        # the remaining case.


def _resolve_type_alias(candidate: Any) -> Any:
    """Unwrap PEP 695 ``type`` aliases down to their underlying type.

    ``pint._typing.Magnitude`` is defined as ``type Magnitude = Scalar |
    Array``.  ``typing.get_origin/get_args`` do not peek through the
    ``TypeAliasType`` wrapper; call this helper first.
    """
    # Python 3.12+ defines ``typing.TypeAliasType`` for PEP 695 aliases.
    try:
        from typing import TypeAliasType as _TypeAliasType
    except ImportError:
        _TypeAliasType = None

    resolved = candidate
    seen: set[int] = set()
    while _TypeAliasType is not None and isinstance(resolved, _TypeAliasType):
        if id(resolved) in seen:
            break
        seen.add(id(resolved))
        resolved = resolved.__value__

    # Recursively resolve PEP 695 aliases *inside* generic args too,
    # since ``Scalar`` itself is a ``type`` alias.
    origin = get_origin(resolved)
    args = get_args(resolved)
    if origin is not None and args:
        try:
            return origin[tuple(_resolve_type_alias(a) for a in args)]
        except Exception:
            # Some origins (e.g. ``types.UnionType``) may not be
            # subscriptable directly; fall back to the resolved value.
            return resolved
    return resolved


def _candidate_is_array_like(arg: Any) -> bool:
    """Best-effort runtime check whether ``arg`` names an ndarray type."""
    # Direct ``np.ndarray`` reference
    if arg is np.ndarray:
        return True
    # ``np.ndarray[Any, Any]`` — its origin is ``np.ndarray``
    origin = get_origin(arg)
    if origin is np.ndarray:
        return True
    # Fallback: string name check for module-local aliases
    return getattr(arg, "__name__", "") == "ndarray"


# ---------------------------------------------------------------------------
# 4. Python 3.14-specific pickle-bookkeeping guard
# ---------------------------------------------------------------------------


@requires_py314
@requires_numpy2
class TestPython314PickleBookkeeping:
    """Guards for the Python 3.14 / NumPy 2.0 combination.

    Python 3.14 introduced stricter attribute-error handling inside
    ``pickle._loads_from_buffer``, and NumPy 2.0 changed how
    ``ndarray.__reduce_ex__`` reports its own shape.  The nested
    ``Quantity.__reduce__`` → ``ndarray.__reduce_ex__`` call chain must
    not raise on either side.
    """

    def setup_method(self):
        self.ureg = UnitRegistry()

    def test_ndarray_magnitude_reduce_ex_does_not_raise(self):
        arr = np.asarray([1.0, 2.0, 3.0])
        q = self.ureg.Quantity(arr, "m")
        # If Quantity's __reduce__ accidentally flattens to form-0 style
        # ``(type, mag, units)`` then on Python 3.14 + NumPy 2.0 pickle
        # would try ``type(mag, units)`` and produce nonsense.  We
        # therefore assert that even after reduce_ex the restored
        # quantity is numerically identical.
        restored = pickle.loads(pickle.dumps(q))
        np.testing.assert_array_equal(restored.magnitude, arr)
        assert restored.units == q.units
