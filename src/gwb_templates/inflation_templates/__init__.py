"""Inflationary gravitational-wave background templates."""

from gwb_templates.inflation_templates.axion_inflation import AxionInflationU1
from gwb_templates.inflation_templates.axion_inflation_matern52 import (
    AxionInflationU1Matern52,
)

__all__ = ["AxionInflationU1", "AxionInflationU1Matern52"]
