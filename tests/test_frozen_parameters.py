import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from gwb_templates import constants as c
from gwb_templates.astrophysical_templates.extragalactic_sobbh_bns import (
    ExtragalacticSobbhBns,
    ExtragalacticSobbhBnsA,
)
from gwb_templates.astrophysical_templates.extragalactic_wd2 import ExtragalacticWd2A
from gwb_templates.astrophysical_templates.galactic_binaries import (
    GalacticBinariesA,
    _fiducial_shape,
)
from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import (
    AnalyticTemplate,
    FrozenParametersMixin,
    Template,
    get_template_from_registry,
)

fvec = jnp.geomspace(c.f_min, c.f_max, 50)

# Every registered variant built with the mixin, so new ones are covered automatically.
FROZEN_VARIANTS = sorted(
    name
    for name, cls in Template.registered_templates().items()
    if issubclass(cls, FrozenParametersMixin) and "PARENT" in cls.__dict__
)


def free_parameters(model):
    """A valid free-parameter vector: the parent's prior midpoints."""
    return jnp.array(
        [
            0.5 * (model.prior_by_param[n]["min"] + model.prior_by_param[n]["max"])
            for n in model.parameter_names
        ]
    )


def full_parameters(model, theta):
    free = iter(theta)
    return [
        getattr(model, n) if n in model.FROZEN else next(free)
        for n in model.parent.parameter_names
    ]


class TestFrozenVariants(unittest.TestCase):

    def test_variants_found(self):
        self.assertGreaterEqual(len(FROZEN_VARIANTS), 7)

    def test_mixin_is_not_registered(self):
        self.assertNotIn("FrozenParametersMixin", Template.registered_templates())

    def test_parameter_names_are_parent_minus_frozen(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                expected = tuple(
                    n for n in model.parent.parameter_names if n not in model.FROZEN
                )
                self.assertEqual(model.parameter_names, expected)

    def test_value_matches_parent(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                theta = free_parameters(model)
                expected = model.parent.omega_gw_h2(
                    fvec, *full_parameters(model, theta)
                )
                np.testing.assert_array_equal(model.omega_gw_h2(fvec, *theta), expected)

    def test_keyword_call_matches_positional(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                theta = free_parameters(model)
                kwargs = dict(zip(model.parameter_names, theta))
                np.testing.assert_array_equal(
                    model.omega_gw_h2(fvec, **kwargs), model.omega_gw_h2(fvec, *theta)
                )

    def test_gradient_vs_autodiff(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                theta = free_parameters(model)
                grad = model.grad_theta_omega_gw_h2(fvec, theta)
                grad_ad = jax.jacfwd(model._omega_from_parameter_vector, argnums=1)(
                    fvec, theta
                )
                self.assertEqual(grad.shape, (fvec.size, model.n_params))
                np.testing.assert_allclose(grad, grad_ad, rtol=1e-10, atol=0.0)

    def test_scalar_frequency(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                grad = model.grad_theta_omega_gw_h2(
                    jnp.array(1e-3), free_parameters(model)
                )
                self.assertEqual(grad.shape, (model.n_params,))

    def test_labels_and_priors_cover_free_parameters(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                self.assertEqual(
                    set(model.parameter_labels), set(model.parameter_names)
                )
                self.assertEqual(set(model.prior_by_param), set(model.parameter_names))


class TestFrozenParametersBehaviour(unittest.TestCase):

    def test_frozen_value_is_read_at_call_time(self):
        model = ExtragalacticSobbhBnsA()
        before = model.omega_gw_h2(fvec, -10.0)
        model.tilt = jnp.array(1.0)
        self.assertFalse(np.array_equal(before, model.omega_gw_h2(fvec, -10.0)))
        self.assertEqual(float(model.frozen["tilt"]), 1.0)

    def test_frozen_override_at_construction(self):
        model = ExtragalacticSobbhBnsA(tilt=jnp.array(1.0))
        expected = ExtragalacticSobbhBns().omega_gw_h2(fvec, -10.0, 1.0)
        np.testing.assert_array_equal(model.omega_gw_h2(fvec, -10.0), expected)

    def test_keywords_are_routed_to_frozen_parent_and_template(self):
        model = ExtragalacticWd2A(
            f_low=jnp.array(2e-4), f0=jnp.array(2e-3), model_name="custom"
        )
        self.assertEqual(float(model.f_low), 2e-4)
        self.assertEqual(float(model.parent.f0), 2e-3)
        self.assertEqual(model.model_name, "custom")
        self.assertNotIn("f0", vars(model))

    def test_unknown_keyword_raises(self):
        with self.assertRaises(TypeError):
            ExtragalacticSobbhBnsA(not_a_parameter=1.0)

    def test_galactic_binaries_observing_setup(self):
        default = GalacticBinariesA()
        expected = _fiducial_shape(1.0, 7.0, 6)
        short = GalacticBinariesA(Tobs_yrs=1.0)
        for name, value in expected.items():
            self.assertEqual(float(getattr(short, name)), float(value))
        self.assertNotEqual(float(short.log_frk), float(default.log_frk))
        self.assertEqual(
            float(GalacticBinariesA(alpha=None).alpha), float(default.alpha)
        )
        self.assertEqual(float(GalacticBinariesA(alpha=jnp.array(1.0)).alpha), 1.0)

    def test_parent_attribute_access_is_deprecated(self):
        model = ExtragalacticSobbhBnsA(ref_freq=jnp.array(1e-2))
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(float(model.ref_freq), 1e-2)
        before = model.omega_gw_h2(fvec, -10.0)
        with self.assertWarns(DeprecationWarning):
            model.pivot = jnp.array(2e-2)  # the attribute the spectrum reads
        self.assertEqual(float(model.parent.pivot), 2e-2)
        self.assertFalse(np.array_equal(before, model.omega_gw_h2(fvec, -10.0)))
        self.assertIn("ref_freq", dir(model))
        self.assertTrue(hasattr(ExtragalacticSobbhBnsA, "DEFAULT_REF_FREQ"))

    def test_supported_access_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            model = ExtragalacticSobbhBnsA(tilt=jnp.array(1.0), ref_freq=1e-2)
            model.parent.ref_freq
            model.tilt = jnp.array(0.5)
            model.omega_gw_h2(fvec, -10.0)
            model.grad_theta_omega_gw_h2(fvec, jnp.array([-10.0]))

    def test_positional_arguments_are_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            model = ExtragalacticSobbhBnsA(jnp.array(1e-2), jnp.array(1.0))
        expected = ExtragalacticSobbhBnsA(ref_freq=jnp.array(1e-2), tilt=1.0)
        np.testing.assert_array_equal(
            model.omega_gw_h2(fvec, -10.0), expected.omega_gw_h2(fvec, -10.0)
        )

    def test_positional_argument_errors(self):
        with self.assertRaises(TypeError):
            ExtragalacticSobbhBnsA(1e-2, 1.0, 3.0)
        with self.assertRaises(TypeError):
            ExtragalacticSobbhBnsA(1e-2, ref_freq=1e-2)
        with self.assertRaises(TypeError):
            get_template_from_registry("Amplitude", 1.0)

    def test_isinstance_of_former_base_class(self):
        self.assertIsInstance(ExtragalacticSobbhBnsA(), ExtragalacticSobbhBns)

    def test_unknown_frozen_parameter_raises_at_class_creation(self):
        with self.assertRaises(ValueError):

            class _Bad(FrozenParametersMixin, AnalyticTemplate):  # noqa: F841
                PARENT = PowerLaw
                FROZEN = {"not_a_parameter": 1.0}

        self.assertNotIn("_Bad", Template.registered_templates())


if __name__ == "__main__":
    unittest.main(verbosity=2)
