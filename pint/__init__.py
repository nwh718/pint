"""
pint
~~~~

Pint is Python module/package to define, operate and manipulate
**physical quantities**: the product of a numerical value and a
unit of measurement. It allows arithmetic operations between them
and conversions from and to different units.

:copyright: 2016 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any, Literal

from .delegates.formatter._format_helpers import formatter
from .errors import (  # noqa: F401
    DefinitionSyntaxError,
    DimensionalityError,
    LogarithmicUnitCalculusError,
    OffsetUnitCalculusError,
    PintError,
    RedefinitionError,
    UndefinedUnitError,
    UnitStrippedWarning,
)
from .formatting import register_unit_format
from .registry import ApplicationRegistry, LazyRegistry, UnitRegistry
from .util import logger, pi_theorem  # noqa: F401

Quantity = UnitRegistry.Quantity
Unit = UnitRegistry.Unit
Measurement = UnitRegistry.Measurement
Context = UnitRegistry.Context
Group = UnitRegistry.Group

try:  # pragma: no cover
    __version__ = version("pint")
except Exception:  # pragma: no cover
    __version__ = "unknown"

_DEFAULT_REGISTRY = LazyRegistry()
application_registry = ApplicationRegistry(_DEFAULT_REGISTRY)

_UnpickleKind = Literal["Quantity", "Unit", "Measurement"]


def _unpickle(constructor: Any, *args: Any) -> Any:
    """Rebuild object upon unpickling.
    All units must exist in the application registry.

    Parameters
    ----------
    constructor
    *args

    Returns
    -------
    object

    """
    from pint.util import UnitsContainer

    for arg in args:
        if isinstance(arg, UnitsContainer):
            for name in arg:
                application_registry.parse_units(name)

    return constructor(*args)


def _unpickle_application_registry(
    kind: _UnpickleKind, _cls: Any, *args: Any
) -> Any:
    return _unpickle(getattr(application_registry, kind), *args)


def _unpickle_quantity(cls: Any, *args: Any) -> Any:
    return _unpickle_application_registry("Quantity", cls, *args)


def _unpickle_unit(cls: Any, *args: Any) -> Any:
    return _unpickle_application_registry("Unit", cls, *args)


def _unpickle_measurement(cls: Any, *args: Any) -> Any:
    return _unpickle_application_registry("Measurement", cls, *args)


def set_application_registry(
    registry: ApplicationRegistry | LazyRegistry | UnitRegistry,
) -> None:
    """Set the application registry, which is used for unpickling operations
    and when invoking pint.Quantity or pint.Unit directly.

    Parameters
    ----------
    registry : pint.UnitRegistry
    """
    application_registry.set(registry)


def get_application_registry() -> ApplicationRegistry:
    """Return the application registry. If :func:`set_application_registry` was never
    invoked, return a registry built using :file:`defaults_en.txt` embedded in the pint
    package.

    Returns
    -------
    pint.UnitRegistry
    """
    return application_registry


__all__ = (
    "Measurement",
    "Quantity",
    "Unit",
    "UnitRegistry",
    "PintError",
    "DefinitionSyntaxError",
    "LogarithmicUnitCalculusError",
    "DimensionalityError",
    "OffsetUnitCalculusError",
    "RedefinitionError",
    "UndefinedUnitError",
    "UnitStrippedWarning",
    "formatter",
    "get_application_registry",
    "set_application_registry",
    "register_unit_format",
    "pi_theorem",
    "__version__",
    "Context",
)
