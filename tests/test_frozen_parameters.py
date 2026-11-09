import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gwb_templates import constants as c
from gwb_templates.astrophysical_templates.extragalactic_sobbh_bns import (
    ExtragalacticSobbhBns,
    ExtragalacticSobbhBnsA,
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
        for n in model._parent.parameter_names
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
                    n for n in model._parent.parameter_names if n not in model.FROZEN
                )
                self.assertEqual(model.parameter_names, expected)

    def test_value_matches_parent(self):
        for name in FROZEN_VARIANTS:
            with self.subTest(name=name):
                model = get_template_from_registry(name)
                theta = free_parameters(model)
                expected = model._parent.omega_gw_h2(
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

    def test_parent_attributes_are_forwarded(self):
        model = ExtragalacticSobbhBnsA(ref_freq=jnp.array(1e-2))
        self.assertEqual(float(model.ref_freq), 1e-2)
        self.assertIn("ref_freq", dir(model))
        self.assertTrue(hasattr(ExtragalacticSobbhBnsA, "DEFAULT_REF_FREQ"))

    def test_isinstance_of_former_base_class(self):
        self.assertIsInstance(ExtragalacticSobbhBnsA(), ExtragalacticSobbhBns)

    def test_unknown_frozen_override_raises(self):
        with self.assertRaises(ValueError):
            get_template_from_registry("Amplitude", frozen={"not_a_parameter": 1.0})

    def test_unknown_frozen_parameter_raises_at_class_creation(self):
        with self.assertRaises(ValueError):

            class _Bad(FrozenParametersMixin, AnalyticTemplate):  # noqa: F841
                PARENT = PowerLaw
                FROZEN = {"not_a_parameter": 1.0}

        self.assertNotIn("_Bad", Template.registered_templates())


if __name__ == "__main__":
    unittest.main(verbosity=2)
