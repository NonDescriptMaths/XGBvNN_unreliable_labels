from functools import partial
from xgboost.sklearn import XGBRegressor
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable

def jax_sle_loss(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate the Squared Log Error loss."""
    return (1/2 * (jnp.log1p(y_pred) - jnp.log1p(y_true))**2)

def hvp(f, inputs, vectors):
    """Hessian-vector product."""
    return jax.jvp(jax.grad(f), inputs, vectors)[1]

def jax_autodiff_grad_hess(
    loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y_true: np.ndarray, y_pred: np.ndarray
):
    """Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`."""
    loss_function_sum = lambda y_pred: loss_function(y_true, y_pred).sum()

    grad_fn = jax.grad(loss_function_sum)
    grad = grad_fn(y_pred)

    hess = hvp(loss_function_sum, (y_pred,), (jnp.ones_like(y_pred), ))

    return grad, hess

# Create xgb model
def get_xgb():
    jax_objective = jax.jit(partial(jax_autodiff_grad_hess, jax_sle_loss))
    return XGBRegressor(objective=jax_objective, n_estimators=100)