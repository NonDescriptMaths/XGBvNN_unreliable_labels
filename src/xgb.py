from functools import partial
from xgboost.sklearn import XGBClassifier
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable
from optax.losses import sigmoid_binary_cross_entropy


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
def get_xgb(lr=0.1, loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = sigmoid_binary_cross_entropy):
    jax_objective = jax.jit(partial(jax_autodiff_grad_hess, loss_function))
    return XGBClassifier(objective=jax_objective, n_estimators=100, max_depth=2, learning_rate=lr)