<p align="center">
  <img src="GWB_templates.png" alt="GWB_templates logo" width="400"/>
</p>

# GWB_templates

**Gravitational Wave Background signal templates for GWB data analysis.**

`gwb_templates` is a Python library providing a unified registry of $h^2\Omega_\mathrm{GW}(f)$ spectral templates, each fully differentiable. Every model exposes a consistent `Signal_model` interface with analytical (or [JAX](https://github.com/google/jax)-computed) first derivatives, making it suitable for gradient-based inference pipelines.

---

## Features

- **JAX-differentiable** — all templates run under `jax.jit`, `jax.grad`, and `jax.vmap`
- **Analytical Jacobians** — hand-derived `dtemplate` for performance-critical models (cosmic strings, resonant/sharp/flat features, …)
- **Unified registry** — look up any model by string label via `get_template`
- **Double precision** — `jax_enable_x64=True` set globally at import time
- **Precomputed data grids** — interpolation-based templates (cosmic strings, resonant features) load data at import and expose JAX-compatible callables

---

## Installation

```bash
pip install .
```

For development (includes linting and testing tools):

```bash
pip install -e ".[dev]"
```

**Dependencies:** `jax>=0.4`, `numpy>=1.21`, `scipy>=1.7`, `interpax>=0.1.0`

---

## Quick Start

```python
import numpy as np
from gwb_templates.templates import get_template

freq = np.geomspace(1e-4, 1e-1, 100)  # Hz

# Retrieve any model by label
model = get_template("power_law")

# Evaluate the spectrum
omega = model.template(freq, pars=[1e-10, 2/3])

# Evaluate the Jacobian  d(Omega)/d(pars),  shape (n_freq, n_pars)
jac   = model.dtemplate(freq, pars=[1e-10, 2/3])
```

`Signal_model` attributes:

| Attribute                  | Description                          |
| -------------------------- | ------------------------------------ |
| `template(freq, pars)`   | $h^2\Omega_\mathrm{GW}(f)$         |
| `dtemplate(freq, pars)`  | Jacobian, shape `(n_freq, n_pars)` |
| `d2template(freq, pars)` | Hessian (where implemented)          |
| `parameter_names`        | List of parameter name strings       |
| `parameter_labels`       | List of LaTeX label strings          |
| `prior`                  | Default prior bounds dict            |

---

## Template Catalogue

### Generic

| Label                                 | Parameters                                                                 | Description                           |
| ------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------- |
| `amplitude`                         | `log_amplitude`                                                                        | Flat amplitude                        |
| `power_law`                         | `log_amplitude, n`                                                                     | Power law A (f/f_*)^n                 |
| `lognormal_bump`                    | `log_amplitude, f_c, σ`                                                                | Log-normal bump                       |
| `broken_power_law`                  | `log_amplitude, f_b, n_1, n_2, Δ`                                                      | Broken power law with free smoothness |
| `broken_power_law_fixed_smoothness` | `log_amplitude, f_b, n_1, n_2`                                                         | Broken power law, fixed smoothness    |
| `broken_power_law_a1`               | `log_amplitude, f_b, n_2`                                                              | Broken power law with fixed low-frequency tilt `n_1=3` |
| `double_broken_power_law`           | `log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2`               | Double broken power law (8 parameters) |
| `double_broken_power_law_rf`        | `log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2`               | Double broken power law (reference-frame variant) |
| `two_double_broken_power_laws`      | `log_amp_1, log_r_amp_2, log_f_12, log_r_f_12, log_r_f_21, log_r_f_22, n_11, n_12, n_13, a_11, a_12, n_21, n_22, n_23, a_21, a_22` | Sum of two double broken power laws (16 parameters) |

### First-Order Phase Transitions

| Label                            | Parameters                     | Description                                             |
| -------------------------------- | ------------------------------ | ------------------------------------------------------- |
| `fopt_broken_power_law`        | `log_amplitude, f_p, n_1, n_2`           | FOPT broken power law                                   |
| `fopt_old`                     | `log_amplitude, f_p, n_1, n_2`           | Legacy FOPT broken power law                            |
| `pt_sound_waves`               | `log_amplitude, f_p`                     | Sound wave contribution (fixed spectral shape)          |
| `pt_turbulence`                | `log_amplitude, f_p`                     | Turbulence contribution (fixed spectral shape)          |
| `pt_collision`                 | `log_amplitude, f_p`                     | Bubble collision contribution                           |
| `pt_plasma`                    | `log_amplitude, f_p`                     | Plasma contribution                                     |

### Inflation

| Label                        | Parameters                  | Description                                             |
| ---------------------------- | --------------------------- | ------------------------------------------------------- |
| `double_peak`              | `log_amplitude, f_1, f_2, σ_1, σ_2` | Double log-normal peak                                  |
| `double_peak_sharp`        | `log_amplitude, f_1, f_2, n_1, n_2`   | Double sharp peak                                       |
| `double_peak_sharp_log`    | log-space params            | `double_peak_sharp` in log parametrisation            |
| `excited_states`           | `log_amplitude, f_c, σ, δ`          | Excited initial states                                  |
| `flat_resonant`            | `log_amplitude, ω`                   | Flat-spectrum resonant modulation                       |
| `flat_resonant_log`        | log-space params            | `flat_resonant` in log parametrisation                |
| `lognormal_bump_sharp`     | `log_amplitude, f_c, σ`              | Log-normal bump with sharp cutoff                       |
| `lognormal_bump_sharp_log` | log-space params            | `lognormal_bump_sharp` in log parametrisation         |
| `sharp_feature`            | `log_amplitude, f_c, ω, φ`          | Sharp-feature oscillatory template (arXiv:1407.4034)    |
| `sharp_feature_log`        | log-space params            | `sharp_feature` in log parametrisation                |
| `resonant_feature`         | `log_amplitude, ω, φ`               | Resonant-feature oscillatory template (arXiv:1407.4034) |
| `resonant_feature_log`     | log-space params            | `resonant_feature` in log parametrisation             |

### Astrophysical Foregrounds

| Label                         | Parameters                      | Description                                          |
| ----------------------------- | ------------------------------- | ---------------------------------------------------- |
| `galactic_binaries`         | `log_amplitude, α, f_k, δ`              | Galactic binary confusion noise                      |
| `galactic_binaries_A`       | `log_amplitude`                           | Galactic binaries — amplitude only (fiducial shape) |
| `galactic_binaries_old`     | `log_amplitude, α, f_k, δ`              | Legacy galactic binary template                      |
| `galactic_binaries_old_A`   | `log_amplitude`                           | Legacy galactic binaries — amplitude only           |
| `extragalactic_sobbh_bns`   | `log_amplitude, tilt`                | Stellar-origin BBH + BNS foreground                  |
| `extragalactic_sobbh_bns_A` | `log_amplitude`                           | SO-BBH/BNS — amplitude only                         |
| `extragalactic_wd`          | `log_amplitude, f_k, δ, α_1, α_2`      | Extragalactic WD binary foreground (Model I)         |
| `extragalactic_wd_A`        | `log_amplitude`                           | Extragalactic WD — amplitude only (fiducial shape)  |
| `extragalactic_wd2`         | `log_amplitude, f_lo, f_hi, α_lo, α_hi` | Extragalactic WD binary foreground (Model II)        |
| `extragalactic_wd2_A`       | `log_amplitude`                           | Extragalactic WD2 — amplitude only (fiducial shape) |

### Cosmic Strings

| Label                         | Parameters                      | Description                                                   |
| ----------------------------- | ------------------------------- | ------------------------------------------------------------- |
| `cosmic_string_model_i`     | `log₁₀(Gμ)`                | Nambu-Goto Model I (BOS, analytical spectrum)                 |
| `cosmic_string_model_i_edf` | `log₁₀(Gμ)`                | Model I with energy-density fraction parametrisation          |
| `cosmic_string_model_i_eos` | `log₁₀(Gμ)`                | Model I with equation-of-state correction                     |
| `cosmic_string_model_ii`    | `log₁₀(Gμ)`                | Model II (BOS$P_n$, precomputed 2-D grid, arXiv:1909.00819) |
| `abelian_higgs_model_ii`    | `log₁₀(Gμ), log₁₀(f_NG)` | Abelian Higgs Model II (amplitude-scaled)                     |

---

## Project Structure

```
src/gwb_templates/
├── templates.py                  # Registry & get_template()
├── utils.py                      # Signal_model dataclass, gradient helpers
├── constants.py                  # Cosmological & LISA constants
├── generic_templates/
├── FOPT_templates/
├── inflation_templates/
├── astrophysical_templates/
└── cosmic_string_templates/
```

---

## Running the Tests

```bash
pytest 
```

---

## License

MIT © Mauro Pieroni
