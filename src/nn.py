import numpy as np
import jax 
import jax.numpy as jnp
import flax.linen as nn
from optax import adam, sgd
from optax.losses import sigmoid_binary_cross_entropy
from typing import Sequence

losses_str2fn = {
    'cross_entropy': sigmoid_binary_cross_entropy
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
    def __init__(self, lr=0.1, opt='adam', loss='cross_entropy', num_epochs=10, batch_size=512, update_ratio=0.5, num_update_epochs=1, MLP_shape=[52,52,1]):
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

        self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, MLP_shape[0])))
        self.opt_state = self.opt.init(self.params)

        self.loss_grad_fn = jax.value_and_grad(self.loss)

    def fit(self, X, y, eval_metric="logloss", eval_set=None):
        for epoch in range(self.num_epochs):
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                loss_val, grads = self.loss_grad_fn(self.model.apply(self.params, X_batch), y_batch)
                
                updates, self.params = self.opt.update(grads, self.params)

                self.params = self.opt.apply_updates(self.params, updates)

                print(f"Epoch {epoch}, Batch {i//self.batch_size}, Loss: {loss_val}")

    def update(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass


if __name__ == '__main__':
    from naive_data import naive_get_data
    X, y, X_test, y_test = naive_get_data()
    X = X.to_numpy()
    y = y.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    model = get_nn()
    model.fit(X, y)
    # Test pretraining
    # results = pretrain(model, X, y, X_test, y_test)
    # print(results)
    # Test training
    # results  =train(model, X, y, X_test, y_test, num_epochs=3)
    # print(results)