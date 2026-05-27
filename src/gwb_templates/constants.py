# ── Cosmological constants (Planck 2018, arXiv:1807.06209) ────────────────────
# Dimensionless Hubble parameter
h = 0.6736
# This factor conversion H0 to SI units (times 100)
Hubble_over_h_SI = 3.24e-18
# Hubble rate in Hz (H0 = h * 100 km/s/Mpc)
H0_SI = h * Hubble_over_h_SI
# Radiation density parameter (photons + 3 SM neutrinos)
Omega_R = 9.1476e-5
# Matter density parameter
Omega_M = 0.3153
# CMB temperature today [K]
T0_CMB = 2.7255
# Scale factor at matter–radiation equality  (a_today ≡ 1)
a_eq = Omega_R / Omega_M
# Effective relativistic degrees of freedom today (photons + 3 light neutrinos)
g_star_0 = 3.91
# Effective dof at high temperature (Standard Model, T >> 100 GeV)
g_star_high_T = 106.75
# CMB temperature today in GeV  (kB * 2.7255 K → GeV)
T0_CMB_GeV = 2.349e-13
# Planck's constant in eV*s
h_bar_eV_s = 6.58e-16
# Planck time in eV^-1
time_planck_eV = 8.19148936e-29
# Planck time [s]
t_planck_SI = h_bar_eV_s * time_planck_eV
# H0 in Planck units  (H0 * h_bar_eV_s)
H0_eV = H0_SI * h_bar_eV_s
# H0 in Planck units  (H0 * t_pl)
H0_natural_units = H0_SI * t_planck_SI

# ── LISA band frequency limits ────────────────────────────────────────────────
# Canonical frequency window used throughout template tests
f_min = 3e-5  # Hz  (lower bound)
f_max = 5e-1  # Hz  (upper bound)
