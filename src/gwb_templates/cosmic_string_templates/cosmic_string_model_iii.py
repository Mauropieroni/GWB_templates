"""
Cosmic String Model III (1 parameter).

Subclasses :class:`CosmicStringModelII`: same 2D (log_Gmu, log10_frequency) grid
layout, bilinear interpolation, and analytical gradient, just pointed at the LRS
cusp-dominated grid file instead of the BOS one, with its own label and citations.

Reference: arXiv:astro-ph/0511646 (Ringeval, Sakellariadou & Bouchet — Nambu-Goto
simulations);
           arXiv:1006.0931 (Lorenz, Ringeval & Sakellariadou — LRS Loop distribution);
           arXiv:1903.06685 (Auclair et al. — Extension of the model).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from gwb_templates.cosmic_string_templates.cosmic_string_model_ii import (
    CosmicStringModelII,
)

_DEFAULT_DATA_FILENAME = "Model-III_LRS-loggrid.dat"


class CosmicStringModelIII(CosmicStringModelII):
    r"""
    Cosmic String Model III (arXiv:1006.0931, Cusp dominated :math:`P_n`).

    1-parameter model evaluated from a precomputed data grid via bilinear
    interpolation. Shares its grid loading, ``setup()``, ``omega_gw_h2``, and
    analytical gradient with :class:`CosmicStringModelII` (same (log_Gmu,
    log10_frequency) grid layout); only the default data file, model label, and
    citations differ here.

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`. Grid range
        :math:`-18 \le \log G\mu \le -9.5`.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Ringeval:2005kr,
    author = "Ringeval, Christophe and Sakellariadou, Mairi and Bouchet, Francois",
    title = "{Cosmological evolution of cosmic string loops}",
    eprint = "astro-ph/0511646",
    archivePrefix = "arXiv",
    doi = "10.1088/1475-7516/2007/02/023",
    journal = "JCAP",
    volume = "02",
    pages = "023",
    year = "2007"
}
""",
        r"""
@article{Lorenz:2010sm,
    author = "Lorenz, Larissa and Ringeval, Christophe and Sakellariadou, Mairi",
    title = "{Cosmic string loop distribution on all length scales and at any redshift
        }",
    eprint = "1006.0931",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2010/10/003",
    journal = "JCAP",
    volume = "10",
    pages = "003",
    year = "2010"
}
""",
        r"""
@article{Auclair:2019zoz,
    author = "Auclair, Pierre and Ringeval, Christophe and Sakellariadou, Mairi and
    Steer, Daniele",
    title = "{Cosmic string loop production functions}",
    eprint = "1903.06685",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "KCL-PH-TH/2019-19",
    doi = "10.1088/1475-7516/2019/06/015",
    journal = "JCAP",
    volume = "06",
    pages = "015",
    year = "2019"
}
""",
    )

    def __init__(
        self,
        data_filename: str = _DEFAULT_DATA_FILENAME,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            data_filename: Filename of the precomputed grid inside the
                ``cosmic_string_templates/data`` directory.
        """
        super().__init__(
            data_filename,
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Cosmic String Model III"
            ),
            parameter_labels=parameter_labels,
            prior_by_param=prior_by_param,
        )
