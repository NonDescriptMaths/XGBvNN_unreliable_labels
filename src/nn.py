import numpy as np
import jax 
import jax.numpy as jnp
import flax.linen as nn
import optax
from optax import adam, sgd
from optax.losses import sigmoid_binary_cross_entropy
from typing import Sequence

losses_str2fn = {
    'cross_entropy': lambda *args: sigmoid_binary_cross_entropy(*args).mean()
}

optimizers_str2fn = {
    'adam': adam,
    'sgd': sgd
}

def get_nn(**kwargs):
    return NNWrapper(**kwargs) # use default parameters unless otherwise specified

class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


class NNWrapper:
    def __init__(self, lr=0.1, opt='adam', loss='cross_entropy', num_epochs=10, batch_size=512, update_ratio=0.5, num_update_epochs=1, MLP_shape=[51,128,1]):
        '''
        lr: learning rate
        opt: optimizer
        loss: loss function
        num_epochs: number of epochs
        batch_size: batch size
        update_ratio: ratio 
        num_update_epochs: number of epochs to update
        '''
        
        self.lr = lr
        self.opt = optimizers_str2fn[opt](learning_rate=lr)
        self.loss = losses_str2fn[loss]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.update_ratio = update_ratio
        self.num_update_epochs = num_update_epochs

        self.model = MLP(features=MLP_shape)

        self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones((MLP_shape[0],)))
        self.opt_state = self.opt.init(self.params)

        self.eval = None

    def loss_grad_fn(self, X, y):

        def loss(params):
            return self.loss(self.model.apply(params, X), y)
        
        return jax.value_and_grad(loss)(self.params)
    
    def one_epoch(self, X, y):
        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]

            loss_val, grads = self.loss_grad_fn(X_batch, y_batch)

            updates, self.opt_state = self.opt.update(grads, self.opt_state)

            self.params = optax.apply_updates(self.params, updates)

    def fit(self, X, y, eval_metric="logloss", eval_set=None):
        for _ in range(self.num_epochs):
            self.one_epoch(X, y)

        if eval_set is not None:
            self.evals = self.validation_loss(eval_metric, eval_set)

    def update(self, X_train, y_train, eval_metric, eval_set, X_prev = None, y_prev = None):
        if X_prev is not None or y_prev is not None:
            # first create new dataset by adding the new data to the old data with a ratio of update_ratio
            idx = np.random.randint(X_prev.shape[0], size=int(X_train.shape[0]*(1-self.update_ratio)))

            X = np.concatenate((X_train, X_prev[idx, :]), axis=0)
            y = np.concatenate((y_train, y_prev[idx, :]), axis=0)

        else:
            X = X_train
            y = y_train

        idx = np.random.permutation(X.shape[0])

        X = X[idx, ...]
        y = y[idx]

        for _ in range(self.num_update_epochs):
            self.one_epoch(X, y)

        if eval_set is not None:
            self.evals = self.validation_loss(eval_metric, eval_set)

    def predict(self, X_test):
        return self.model.apply(self.params, X_test)
    
    def validation_loss(self, eval_metric, eval_set):
        X_val, y_val = eval_set[0]  # TODO: check why eval_set is a single tuple inside a list
        
        loss_val = self.loss(self.model.apply(self.params, X_val), y_val)

        if eval_metric == 'logloss':
            loss_val = jnp.log(loss_val)

        return loss_val
    
    def evals_result(self):
        return self.evals

if __name__ == '__main__':
    # from naive_data import naive_get_data
    # X, y, X_test, y_test = naive_get_data()
    from utils import get_data, get_X_y, get_X_y_labelled
    from train import pretrain, train

    train_set, test_set, validate_set = get_data()
    X, y = get_X_y_labelled(train_set)
    X_test, y_test = get_X_y_labelled(test_set)

    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    # breakpoint()

    model = get_nn()
    model.fit(X, y)

    # Test pretraining
    # results = pretrain(model, X, y, X_test, y_test)
    # print(results)

    # Test training (involves pretraining as well)
    results  = train(model, X, y, X_test, y_test, num_epochs=5)
    print(results)