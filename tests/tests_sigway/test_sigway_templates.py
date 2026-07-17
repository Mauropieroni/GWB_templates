"""
Tests for the SIGWAY-backed templates.

SIGWAY is an optional dependency (and needs a JAX new enough for
``jax.scipy.special.sici``). When it is not installed, ``import gwb_templates``
must still succeed and the SIGWAY templates must not register; the
SIGWAY-specific tests are skipped via ``SIGWAY_AVAILABLE``.
"""

import unittest

import jax.numpy as jnp

import gwb_templates
from gwb_templates import SIGWAY_AVAILABLE
from gwb_templates.template import Template, get_template_from_registry

# The predefined SIGWAY templates and a fiducial parameter set for each
# (matching the configurations of arXiv:2501.11320).
SIGWAY_TEMPLATES = {
    "SIGWAYLognormal": (-2.5, -0.30102999, -2.0),
    "SIGWAYBrokenPowerLaw": (
        -2.71028683,
        3.10673309,
        0.22154827,
        1.25417717,
        -1.58150632,
    ),
    "SIGWAYMultifieldOscillations": (-1.5, -1.5, 0.5, 14.0, 1.0),
    "SIGWAYEarlyMatterDomination": (2.1e-9, 0.06, 2000.0),
    "SIGWAYSingleFieldUSR": (
        0.7122360088466501,
        1.47312e-06,
        0.19688844475447168,
        1.8690166574800342e-05,
    ),
}


class TestSigwayOptionalImport(unittest.TestCase):
    """Behaviour that must hold regardless of whether SIGWAY is installed."""

    def test_flag_is_bool(self):
        self.assertIsInstance(SIGWAY_AVAILABLE, bool)

    def test_registration_matches_availability(self):
        registered = [
            n for n in Template.registered_templates() if n.startswith("SIGWAY")
        ]
        if SIGWAY_AVAILABLE:
            self.assertEqual(set(registered), set(SIGWAY_TEMPLATES))
        else:
            self.assertEqual(registered, [])

    def test_scalar_induced_base_always_available(self):
        # The physics-category marker base is dependency-free: always importable.
        from gwb_templates.scalar_induced_templates.base import (
            ScalarInducedTemplate,
        )

        self.assertTrue(issubclass(ScalarInducedTemplate, Template))
        self.assertTrue(hasattr(gwb_templates, "ScalarInducedTemplate"))

    def test_sigway_symbols_match_availability(self):
        self.assertEqual(
            hasattr(gwb_templates, "sigway_template"), SIGWAY_AVAILABLE
        )
        self.assertEqual(
            hasattr(gwb_templates, "SIGWAYTemplate"), SIGWAY_AVAILABLE
        )


@unittest.skipUnless(SIGWAY_AVAILABLE, "SIGWAY not installed")
class TestSigwayTemplates(unittest.TestCase):
    """End-to-end behaviour of the predefined SIGWAY templates."""

    def test_spectrum_and_gradient_shapes(self):
        fvec = jnp.geomspace(1e-4, 1e-2, 16)
        for name, params in SIGWAY_TEMPLATES.items():
            with self.subTest(template=name):
                model = get_template_from_registry(name)
                self.assertEqual(model.parameter_names, _names(model))
                self.assertEqual(len(params), model.n_params)

                pmap = dict(zip(model.parameter_names, params))
                omega = model.omega_gw_h2_from_parameters(fvec, pmap)
                self.assertEqual(omega.shape, fvec.shape)

                grad = model.grad_theta_omega_gw_h2(fvec, pmap)
                self.assertEqual(grad.shape, (fvec.shape[0], model.n_params))

    def test_bibtex_present(self):
        for name in SIGWAY_TEMPLATES:
            with self.subTest(template=name):
                bib = get_template_from_registry(name).get_bibtex()
                self.assertIn("2501.11320", bib)

    def test_usr_is_non_jittable_finite_difference(self):
        model = get_template_from_registry("SIGWAYSingleFieldUSR")
        self.assertFalse(model.jittable)
        self.assertEqual(model.differentiation_backend, "finite_difference")

    def test_factory_builds_and_registers(self):
        from sigway.kernels import RadiationKernel

        from gwb_templates.scalar_induced_templates.sigway.base import (
            sigway_template,
        )

        def pz(k, logAs, logks):
            return 10.0**logAs * jnp.exp(-0.5 * (jnp.log(k / 10**logks) / 0.3) ** 2)

        cls = sigway_template(
            "FactoryTestLN",
            pzeta=pz,
            parameter_names=("logAs", "logks"),
            kernel=RadiationKernel(),
            s=jnp.linspace(0.0, 1.0, 10),
            t=jnp.concatenate(
                [jnp.linspace(1e-5, 0.999, 100), jnp.geomspace(1.0, 1e3, 300)]
            ),
            f=jnp.geomspace(1e-4, 1e-2, 50),
            upsample=True,
        )
        try:
            self.assertIn("FactoryTestLN", Template.registered_templates())
            model = cls()
            self.assertEqual(model.parameter_names, ("logAs", "logks"))
            omega = model.omega_gw_h2_from_parameters(
                jnp.geomspace(1e-4, 1e-2, 12), {"logAs": -2.0, "logks": -3.0}
            )
            self.assertEqual(omega.shape, (12,))
        finally:
            Template._registry.pop("FactoryTestLN", None)


def _names(model):
    return tuple(model._model.parameter_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
