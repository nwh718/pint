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

import sys
from importlib.metadata import version

from .delegates.formatter._format_helpers import formatter
from .errors import (  # noqa: F401
    DefinitionSyntaxError,
    DimensionalityError,
    LogarithmicUnitCalculusError,
    OffsetUnitCalculusError,
    PintError,
    PythonVersionError,
    RedefinitionError,
    UndefinedUnitError,
    UnitStrippedWarning,
)
from .formatting import register_unit_format
from .registry import ApplicationRegistry, LazyRegistry, UnitRegistry
from .util import logger, pi_theorem  # noqa: F401

_SUPPORTED_PYTHON_VERSIONS = ((3, 12), (3, 13), (3, 14))
_SUPPORTED_PYTHON_VERSION_TEXT = "3.12, 3.13, and 3.14"


def _check_python_version(version_info: tuple[int, int] | None = None) -> None:
    current = tuple(version_info if version_info is not None else sys.version_info[:2])

    if current in _SUPPORTED_PYTHON_VERSIONS:
        return

    current_text = f"{current[0]}.{current[1]}"

    if current < _SUPPORTED_PYTHON_VERSIONS[0]:
        raise PythonVersionError(
            f"Pint no longer supports Python {current_text}. "
            f"Support for Python 3.11 and older has been dropped. "
            f"Use Python {_SUPPORTED_PYTHON_VERSION_TEXT}."
        )

    raise PythonVersionError(
        f"Pint does not yet support Python {current_text}. "
        f"Use Python {_SUPPORTED_PYTHON_VERSION_TEXT}."
    )


_check_python_version()

# Default Quantity, Unit and Measurement are the ones
# build in the default registry.
Quantity = UnitRegistry.Quantity
Unit = UnitRegistry.Unit
Measurement = UnitRegistry.Measurement
Context = UnitRegistry.Context
Group = UnitRegistry.Group
try:  # pragma: no cover
    __version__ = version("pint")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"

#: A Registry with the default units and constants.
_DEFAULT_REGISTRY = LazyRegistry()

#: Registry used for unpickling operations.
application_registry = ApplicationRegistry(_DEFAULT_REGISTRY)


def _unpickle(cls, *args):
    """Rebuild object upon unpickling.

    All units must exist in the application registry.

    Parameters
    ----------
    cls : Quantity, Magnitude, or Unit
    *args

    Returns
    -------
    object of type cls

    """
    from pint.util import UnitsContainer

    for arg in args:
        if isinstance(arg, UnitsContainer):
            for name in arg:
                application_registry.parse_units(name)

    return cls(*args)



def _unpickle_quantity(cls, *args):
    """Rebuild quantity upon unpickling using the application registry."""
    return _unpickle(application_registry.Quantity, *args)



def _unpickle_unit(cls, *args):
    """Rebuild unit upon unpickling using the application registry."""
    return _unpickle(application_registry.Unit, *args)



def _unpickle_measurement(cls, *args):
    """Rebuild measurement upon unpickling using the application registry."""
    return _unpickle(application_registry.Measurement, *args)



def set_application_registry(registry):
    """Set the application registry, which is used for unpickling operations
    and when invoking pint.Quantity or pint.Unit directly.

    Parameters
    ----------
    registry : pint.UnitRegistry
    """
    application_registry.set(registry)



def get_application_registry():
    """Return the application registry. If :func:`set_application_registry` was never
    invoked, return a registry built using :file:`defaults_en.txt` embedded in the pint
    package.

    Returns
    -------
    pint.UnitRegistry
    """
    return application_registry


__all__ = (
    "ApplicationRegistry",
    "Context",
    "Measurement",
    "PintError",
    "PythonVersionError",
    "Quantity",
    "RedefinitionError",
    "UndefinedUnitError",
    "Unit",
    "UnitRegistry",
    "UnitStrippedWarning",
    "DefinitionSyntaxError",
    "LogarithmicUnitCalculusError",
    "DimensionalityError",
    "OffsetUnitCalculusError",
    "formatter",
    "get_application_registry",
    "set_application_registry",
    "register_unit_format",
    "pi_theorem",
    "__version__",
)
