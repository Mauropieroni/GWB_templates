# Global
import unittest
import jax
import jax.numpy as jnp

# Local
from gwb_templates import constants as c
from gwb_templates.templates import get_template


def not_a_test(object):
    object.__test__ = False
    return object


# Define test parameters for different models
POWER_LAW_PARAMS = jnp.array([-10.0, 2.0])
power_law_model = get_template("power_law")

f_points = 100
fvec = jnp.geomspace(c.f_min, c.f_max, f_points)
pivot = 10 ** jnp.mean(jnp.log10(fvec))


@not_a_test
def test_dmodel(model, parameters, **kwargs):
    """
    Given a signal model and parameters, checks the analytical derivatives
    against the numerical derivatives computed with the forward diff.

    Parameters:
    -----------
    model : Signal_model
        The signal model object to test
    parameters : Array
        Parameters for the signal model
    kwargs : dict
        Additional keyword arguments to pass to the signal model

    Returns:
    --------
    tuple
        Tuple containing the analytical and numerical derivatives
    """

    # Use model's dtemplate method for analytical derivatives
    dtemplate = model.dtemplate(fvec, parameters, **kwargs)

    # Calculate numerical derivatives using forward differences for comparison
    dtemplate_forward = jax.jacfwd(model.template, argnums=1)(
        fvec, parameters, **kwargs
    )

    return dtemplate, dtemplate_forward


class TestSignals(unittest.TestCase):

    def test_dpower_law(self):
        """
        Test function for the first derivative of a power law signal model.
        """
        dtemplate, dtemplate_forward = test_dmodel(
            power_law_model, POWER_LAW_PARAMS, pivot=pivot
        )
        self.assertAlmostEqual(
            jnp.sum(dtemplate - dtemplate_forward).item(), 0.0, places=15
        )

    def test_hessian_power_law(self):
        """
        Test function for the second derivative of a power law signal model.
        """

        model = power_law_model.template(fvec, POWER_LAW_PARAMS, pivot=pivot)

        hessian = power_law_model.d2template(fvec, POWER_LAW_PARAMS, pivot=pivot)

        hessian_analytic = model[None, None, :] * jnp.array(
            [
                [
                    jnp.log(10.0) ** 2 * fvec**0,
                    jnp.log(10.0) * jnp.log(fvec / pivot),
                ],
                [
                    jnp.log(10.0) * jnp.log(fvec / pivot),
                    jnp.log(fvec / pivot) ** 2,
                ],
            ]
        )

        diff_sum = jnp.sum(jnp.abs(hessian - hessian_analytic.T))
        self.assertAlmostEqual(diff_sum.item(), 0.0, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
