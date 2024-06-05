import numpy as np
import jax 
import jax.numpy as jnp
import flax.linen as nn
from  flax.training import train_state
import optax
from optax import adam, sgd
from optax.losses import sigmoid_binary_cross_entropy
from typing import Sequence
from sklearn.metrics import roc_auc_score, accuracy_score


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

@jax.jit
def train_step(state, X_batch, y_batch):

    def loss(params):
        logits = state.apply_fn(params, X_batch).flatten()

        # return loss_fn(logits, y_batch)
        return jnp.mean(sigmoid_binary_cross_entropy(logits, y_batch)), logits
    
    loss_grad_fn = jax.value_and_grad(loss, has_aux=True)
    (loss_val, logits), grads = loss_grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    # updates, self.opt_state = self.opt.update(grads, self.opt_state)
    # self.params = optax.apply_updates(self.params, updates)

    # print(loss_val)

    return state, loss_val, logits

@jax.jit
def eval_step(state, X_batch, y_batch):
    logits = state.apply_fn(state.params, X_batch).flatten()
    loss_val = sigmoid_binary_cross_entropy(logits, y_batch)

    return loss_val, logits


class NNWrapper:
    def __init__(self, lr=0.05, opt='adam', loss='cross_entropy', num_epochs=10, batch_size=512, update_ratio=0.5, num_update_epochs=1, MLP_shape=[51,128,128,1]):
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
        # self.loss = losses_str2fn[loss]
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.update_ratio = update_ratio
        self.num_update_epochs = num_update_epochs

        self.model = MLP(features=MLP_shape)

        self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones((MLP_shape[0],)))
        self.opt_state = self.opt.init(self.params)

        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=self.params, tx=self.opt)

        self.metrics = {'loss': {'training': [], 'validation': []},
                        'auc' : {'training': [], 'validation': []},
                        'accuracy' : {'training': [], 'validation': []},
                        'tpr' : {'training': [], 'validation': []},}
        
    def update_metrics(self, loss, logits, y, set):
        self.metrics['loss'][set].append(loss)

        y_pred = jnp.where(jax.nn.sigmoid(logits) > 0.5, 1, 0)
        # self.metrics['auc'][set].append(roc_auc_score(y, y_pred))
        self.metrics['accuracy'][set].append(accuracy_score(y, y_pred))
        self.metrics['tpr'][set].append(y_pred[y == 1].mean())

    def one_epoch(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]

            # loss_val, grads = self.loss_grad_fn(X_batch, y_batch)

            self.state, loss_val, logits = train_step(self.state, X_batch, y_batch)

            self.update_metrics(loss_val, logits, y_batch, 'training')

    def fit(self, X, y, eval_metric="logloss", eval_set=None):
        for _ in range(self.num_epochs):
            self.one_epoch(X, y)

        if eval_set is not None:
            self.validation(eval_metric, eval_set)

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
            self.validation(eval_metric, eval_set)

    def predict(self, X_test):
        return self.model.apply(self.params, X_test)
    
    def validation(self, eval_metric, eval_set):
        X_val, y_val = eval_set[0]  # TODO: check why eval_set is a single tuple inside a list
        
        loss_val = []
        for i in range(0, X_val.shape[0], self.batch_size):
            X_batch = X_val[i:i+self.batch_size]
            y_batch = y_val[i:i+self.batch_size]

            loss, logits = eval_step(self.state, X_batch, y_batch)
            self.update_metrics(loss.mean(), logits, y_batch, 'validation')

            loss_val.append(loss)

        # breakpoint()

        loss_val = jnp.concat(loss_val, axis=0).mean()
        if eval_metric == 'logloss':
            loss_val = jnp.log(loss_val)

        return loss_val
    
    def get_metrics(self, type='loss', set='validation', last_only=False):
        metric = self.metrics[type][set]
        if last_only:
            return metric[-1]
        return metric
    
    def evals_result(self):
        return self.get_metrics(type='loss', set='validation', last_only=True)
    
if __name__ == '__main__':
    # from naive_data import naive_get_data
    # X, y, X_test, y_test = naive_get_data()
    from utils import get_data, get_X_y, get_X_y_labelled
    from train import pretrain, train

    FULL_SET = True
    FULL_SET_PROP = 0.3
    NUM_EPOCHS = 25


    train_set, test_set, validate_set = get_data()
    if FULL_SET:
        X, y = get_X_y(train_set, true_labels=True, prop=FULL_SET_PROP)
        X_test, y_test = get_X_y(test_set, true_labels=True, prop=FULL_SET_PROP)
    else:
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
    results = train(model, X, y, X_test, y_test, num_epochs=NUM_EPOCHS)
    print(results)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(model.get_metrics(type='loss', set='training'), label='Training')
    ax[0].plot(model.get_metrics(type='loss', set='validation'), label='Validation')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(model.get_metrics(type='accuracy', set='training'), label='Training')
    ax[1].plot(model.get_metrics(type='accuracy', set='validation'), label='Validation')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    ax[2].plot(model.get_metrics(type='tpr', set='training'), label='Training')
    ax[2].plot(model.get_metrics(type='tpr', set='validation'), label='Validation')
    ax[2].set_title('TPR')
    ax[2].legend()

    plt.savefig('../plots/nn_metrics.png')

    breakpoint()
