from functools import partial
from xgboost.sklearn import XGBClassifier
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable
from optax.losses import sigmoid_binary_cross_entropy

# Wrapper for the xgb model that implements an update method
class XGBWrapper:
    def __init__(self, model):
        self.model = model

    def update(self, X, y, eval_metric="logloss", eval_set=None):
        return self.model.fit(X, y, eval_metric=eval_metric, eval_set=eval_set, xgb_model=self.model)

    def predict(self, X):
        return self.model.predict(X)
    
    def fit(self, X, y, eval_metric="logloss", eval_set=None):
        return self.model.fit(X, y, eval_metric=eval_metric, eval_set=eval_set)
    
    def evals_result(self):
        return self.model.evals_result()


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
    return XGBWrapper(XGBClassifier(objective=jax_objective, n_estimators=100, max_depth=2, learning_rate=lr))