from __future__ import annotations

from dataclasses import dataclass

from ._typing import Magnitude
from .converters import Converter


@dataclass(frozen=True)
class ReverseConverter(Converter):
    def convert(self, value: Magnitude, inverse: bool = False) -> Magnitude:
        if inverse:
            return -value
        return value