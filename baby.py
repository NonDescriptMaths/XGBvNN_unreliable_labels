import numpy as np
from xgboost.sklearn import XGBRegressor
import jax
import jax.numpy as jnp
from typing import Callable
from functools import partial
import pandas as pd
from datasets import Dataset

# Load in data 
def get_data():
    card_df = pd.read_csv('/kaggle/input/bank-account-fraud-dataset-neurips-2022/Base.csv')
    card_org = card_df.copy()
    # Reduce the size of the dataset
    card_df = card_df.sample(n = 400000,random_state=42)
    # One hot encode
    card_df = pd.get_dummies(card_df)
    return card_df

# Loss function
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

# Create model
def get_model():
    jax_objective = jax.jit(partial(jax_autodiff_grad_hess, jax_sle_loss))
    return XGBRegressor(objective=jax_objective, n_estimators=100)

# Train model
def train():
    data = get_data()
    model = get_model()
    data = data.to_numpy()
    ds = Dataset.from_dict({"data": data})
    ds = ds.with_format("jax")  
    dl = ds.dataloader(batch_size=32, shuffle=True)
    # Begin Training!
    for epoch in range(num_epochs):
        subkey = np.asarray(jax.random.fold_in(key, epoch))
        ds.shuffle(seed=epoch)
        epoch_loss = []
        for batch_num, batch in enumerate(ds.iter(batch_size=batch_size)):
            batch_x = batch['data']
            X_train, y_train = 
            model.fit(X_train, y_train)

