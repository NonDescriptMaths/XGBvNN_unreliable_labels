from functools import partial
from xgboost.sklearn import XGBClassifier
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable
from optax.losses import sigmoid_binary_cross_entropy

# Wrapper for the xgb model to make it easily swappable with the neural network model in the training loop
class XGBWrapper:
    def __init__(self, model):
        self.model = model
        self. all_metrics = {
            'loss': {'training': [], 'validation': []},
            'auc': {'training': [], 'validation': []},
            'acc': {'training': [], 'validation': []}
        }
    def update(self, X, y, eval_set=None):
        """ Update the model with new data."""
        return self.model.fit(X, y, eval_set=eval_set, xgb_model=self.model)

    def predict(self, X):
        """ Predict on the given data."""
        return self.model.predict(X)
    
    def fit(self, X, y, eval_set=None):
        """ Fit the model on the given data."""
        return self.model.fit(X, y, eval_set=eval_set)
    
    def get_metrics(self):
        """ Get the metrics from the model."""
        results = self.model.evals_result()
        breakpoint()
        self.all_metrics['loss']['training'] += results['validation_0']['logloss']
        self.all_metrics['loss']['validation'] += results['validation_1']['logloss']
        self.all_metrics['auc']['training'] += results['validation_0']['auc']
        self.all_metrics['auc']['validation'] += results['validation_1']['auc']
        self.all_metrics['acc']['training'] += [1-r for r in results['validation_0']['error']]
        self.all_metrics['acc']['validation'] += [1-r for r in results['validation_1']['error']]
        return self.all_metrics

# ----- Currently not using this -----
# def hvp(f, inputs, vectors):
#     """Hessian-vector product."""
#     return jax.jvp(jax.grad(f), inputs, vectors)[1]

# def jax_autodiff_grad_hess(
#     loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
#     y_true: np.ndarray, y_pred: np.ndarray
# ):
#     """Perform automatic differentiation to get the
#     Gradient and the Hessian of `loss_function`."""
#     loss_function_sum = lambda y_pred: loss_function(y_true, y_pred).sum()

#     grad_fn = jax.grad(loss_function_sum)
#     grad = grad_fn(y_pred)

#     hess = hvp(loss_function_sum, (y_pred,), (jnp.ones_like(y_pred), ))

#     return grad, hess
# ----- Currently not using this -----

# Create xgb model
def get_xgb(lr=0.1, n_estimators=100, max_depth=2):
    """ Create an XGBClassifier model with the given learning rate. """
    return XGBWrapper(XGBClassifier(objective='binary:logistic', n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, eval_metric=["logloss", "error", "auc", "aucpr", "pre"]))