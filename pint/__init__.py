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
import warnings
from pint.errors import PythonVersionError

if sys.version_info < (3, 12):
    raise PythonVersionError(
        f"Pint requires Python 3.12, 3.13, or 3.14. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}. "
        f"Python 3.11 is no longer supported."
    )
elif sys.version_info[:2] not in ((3, 12), (3, 13), (3, 14)):
    warnings.warn(
        f"Pint is explicitly tested on Python 3.12, 3.13, and 3.14. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"which may not be fully supported.",
        UserWarning
    )

# Default Quantity, Unit and Measurement are the ones
# build in the default registry.
    cls : Quantity, Magnitude, or Unit
    *args

    Returns
    -------
    object of type cls

    """
    from pint.util import UnitsContainer

    for arg in args:
    """Rebuild unit upon unpickling using the application registry."""
    return _unpickle(application_registry.Unit, *args)


def _unpickle_measurement(cls, *args):
    """Rebuild measurement upon unpickling using the application registry."""
    return _unpickle(application_registry.Measurement, *args)


def set_application_registry(registry):
    """Set the application registry, which is used for unpickling operations
    and when invoking pint.Quantity or pint.Unit directly.
        # Prefixed units are defined within the registry
        # on parsing (which does not happen here).
        # We need to make sure that this happens before using.

    Parameters
    ----------
    registry : pint.UnitRegistry
    """
    application_registry.set(registry)

def get_application_registry():
    """Return the application registry. If :func:`set_application_registry` was never
    invoked, return a registry built using :file:`defaults_en.txt` embedded in the pint
    package.

    -------
    pint.UnitRegistry
    """
    return application_registry

# Enumerate all user-facing objects
# Hint to intersphinx that, when building objects.inv, these objects must be registered
# under the top-level module and not in their original submodules
__all__ = (
    "Measurement",
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
    "set_application_registry",
    "register_unit_format",
    "pi_theorem",
    "__version__",
    "Context",
)
# Enumerate all user-facing objects
# Hint to intersphinx that, when building objects.inv, these objects must be registered
# under the top-level module and not in their original submodules
    "RedefinitionError",
    "UndefinedUnitError",
    "UnitStrippedWarning",
    "Context",
