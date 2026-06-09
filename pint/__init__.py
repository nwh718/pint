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


def _unpickle(cls: type, *args) -> object:
    """Rebuild object upon unpickling.

    Parameters
    ----------
    cls : type
        The class of the object to unpickle.
    *args
        Arguments passed to the class constructor.

    Returns
    -------
    object
        The rebuilt object.
    """
    for name in ("Quantity", "Unit", "Measurement"):
        if name in cls.__name__:
            return getattr(application_registry, name)(*args)
    return cls(*args)


# Keep aliases for backwards compatibility with old pickles
_unpickle_quantity = _unpickle
_unpickle_measurement = _unpickle
_unpickle_unit = _unpickle


def set_application_registry(registry: UnitRegistry) -> None:
    """Set the application registry, which is used for unpickling operations
    and when invoking pint.Quantity or pint.Unit directly.

    Parameters
    ----------
    registry : pint.UnitRegistry
    """
    global application_registry
    application_registry = ApplicationRegistry(registry)


def get_application_registry() -> ApplicationRegistry:
    """Return the application registry. If :func:`set_application_registry` was never
    invoked, return a registry built using :file:`defaults_en.txt` embedded in the pint
    package.

    Returns
    -------
    pint.UnitRegistry
    """
    return application_registry


# Enumerate all user-facing objects
# Hint to intersphinx that, when building objects.inv, these objects must be registered
# under the top-level module and not in their original submodules
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
