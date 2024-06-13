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
from metric import MetricStore


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

    def upweight_positive_loss(loss_val, factor=2):
        return jnp.where(y_batch == 1, factor*loss_val, loss_val)

    def loss(params):
        logits = state.apply_fn(params, X_batch).flatten()

        # return loss_fn(logits, y_batch)
        return jnp.mean(upweight_positive_loss(sigmoid_binary_cross_entropy(logits, y_batch))), logits
        # return jnp.mean(sigmoid_binary_cross_entropy(logits, y_batch)), logits
    
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
    loss_val = jnp.mean(sigmoid_binary_cross_entropy(logits, y_batch))

    return loss_val, logits


class NNWrapper:
    def __init__(self, lr=0.05, opt='adam', loss='cross_entropy', num_epochs=10, batch_size=512, num_update_epochs=10, MLP_shape='128,128'):
        '''
        lr: learning rate
        opt: optimizer
        loss: loss function
        num_epochs: number of epochs
        batch_size: batch size
        num_update_epochs: number of epochs to update
        '''
        
        self.lr = lr
        self.opt = optimizers_str2fn[opt](learning_rate=lr)
        # self.loss = losses_str2fn[loss]
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_update_epochs = num_update_epochs

        MLP_shape = [51, *[int(w) for w in MLP_shape.split(',')],1]

        self.model = MLP(features=MLP_shape)

        self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones((MLP_shape[0],)))
        self.opt_state = self.opt.init(self.params)

        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=self.params, tx=self.opt)

        self.metric_store = MetricStore()


        # self.metrics = {'loss': {'training': [], 'validation': []},
        #                 'auc' : {'training': [], 'validation': []},
        #                 'accuracy': {'training': [], 'validation': []},
        #                 'tpr' : {'training': [], 'validation': []},}
        
        # self.full_metrics = {'loss': {'training': [], 'validation': []},
        #                      'auc' : {'training': [], 'validation': []},
        #                      'accuracy': {'training': [], 'validation': []},
        #                      'tpr' : {'training': [], 'validation': []},}
        
    # def update_metrics(self, loss, logits, y, set, final_epoch=False):
    #     y_pred = jnp.where(jax.nn.sigmoid(logits) > 0.5, 1, 0)

    #     auc = roc_auc_score(y, y_pred)
    #     accuracy = accuracy_score(y, y_pred)
    #     tpr = y_pred[y == 1].mean()
        
    #     self.full_metrics['loss'][set].append(loss)
    #     self.full_metrics['auc'][set].append(auc)
    #     self.full_metrics['accuracy'][set].append(accuracy)
    #     self.full_metrics['tpr'][set].append(tpr)

    #     if final_epoch:
    #         self.metrics['loss'][set].append(loss)
    #         self.metrics['auc'][set].append(auc)
    #         self.metrics['accuracy'][set].append(accuracy)
    #         self.metrics['tpr'][set].append(tpr)
                    


    def one_epoch(self, X, y, final_epoch=False):
        logits_collection = []
        total_loss_val = 0
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i:i+self.batch_size]
            y_batch = y[i:i+self.batch_size]

            # logits = self.model.apply(self.params, X_batch).flatten()

            # make the batch have an equal number of positive and negative samples
            # if y_batch.sum() > 0:
            #     pos_idx = np.where(y_batch == 1)[0]
            #     neg_idx = np.where(y_batch == 0)[0]

            #     if len(pos_idx) > len(neg_idx):
            #         pos_idx = np.random.choice(pos_idx, len(neg_idx), replace=False)
            #     elif len(neg_idx) > len(pos_idx):
            #         neg_idx = np.random.choice(neg_idx, len(pos_idx), replace=False)

            #     idx = np.concatenate((pos_idx, neg_idx))
            #     X_batch = X_batch[idx, :]
            #     y_batch = y_batch[idx]

            # loss_val, grads = self.loss_grad_fn(X_batch, y_batch)

            self.state, loss_val, logits = train_step(self.state, X_batch, y_batch)

            logits_collection.append(logits)
            total_loss_val += loss_val*X_batch.shape[0]

        logits = jnp.concatenate(logits_collection, axis=0)
        total_loss_val /= X.shape[0]

        return total_loss_val, logits
        # self.update_metrics(total_loss_val, logits, y, 'training', final_epoch=final_epoch)

    def fit(self, X, y, eval_set=None):
        for i in range(self.num_epochs):
            total_loss_val, logits = self.one_epoch(X, y, final_epoch=(i == self.num_epochs-1))

        self.metric_store.log({'loss': {'training': total_loss_val}})
        self.metric_store.calculate_metrics(y, jax.nn.sigmoid(logits), 'training')
        

        # if eval_set is not None:
        #     self.validation(eval_metric, eval_set)

        if eval_set is not None:
            eval_set_names = ['training', 'validation', 'test']
            for i, (X_val, y_val) in enumerate(eval_set):
                if i == 0:
                    continue
                total_loss_val, logits = self.validation((X_val, y_val))
    
            self.metric_store.log({'loss': {eval_set_names[i]: total_loss_val}})
            self.metric_store.calculate_metrics(y_val, jax.nn.sigmoid(logits), eval_set_names[i])



    def update(self, X, y, eval_set):
        idx = np.random.permutation(X.shape[0])

        X = X[idx, ...]
        y = y[idx]

        for i in range(self.num_update_epochs):
            total_loss_val, logits = self.one_epoch(X, y, final_epoch=(i == self.num_update_epochs-1))
        
        self.metric_store.log({'loss': {'training': total_loss_val}})
        self.metric_store.calculate_metrics(y, jax.nn.sigmoid(logits), 'training')

        if eval_set is not None:
            eval_set_names = ['training', 'validation', 'test']
            for i, (X_val, y_val) in enumerate(eval_set):
                if i == 0:
                    continue
                total_loss_val, logits = self.validation((X_val, y_val))
    
            self.metric_store.log({'loss': {eval_set_names[i]: total_loss_val}})
            self.metric_store.calculate_metrics(y_val, jax.nn.sigmoid(logits), eval_set_names[i])


    def predict(self, X_test):
        return self.model.apply(self.params, X_test)
    
    def validation(self, eval_set):
        X_val, y_val = eval_set
        
        logits_collection = []
        total_loss_val = 0
        for i in range(0, X_val.shape[0], self.batch_size):
            X_batch = X_val[i:i+self.batch_size]
            y_batch = y_val[i:i+self.batch_size]

            loss_val, logits = eval_step(self.state, X_batch, y_batch)

            logits_collection.append(logits)
            total_loss_val += loss_val*X_batch.shape[0]

        logits = jnp.concatenate(logits_collection, axis=0)
        total_loss_val /= X_val.shape[0]

        # self.update_metrics(total_loss_val, logits, y_val, 'validation', final_epoch=True)

        # return log-loss if specified
        # if eval_metric == 'logloss':
        total_loss_val = -jnp.log(total_loss_val)

        return total_loss_val, logits
    
    def get_metrics(self, type='loss', set='validation', log_final=False):
        return self.metric_store
    #     return self.metrics[type][set]
    
    # def get_full_metrics(self, type='loss', set='validation'):
    #     return self.full_metrics[type][set]
    
    # def evals_result(self):
    #     return self.get_metrics(type='loss', set='validation')
    
if __name__ == '__main__':
    # from naive_data import naive_get_data
    # X, y, X_test, y_test = naive_get_data()
    from utils import get_data, get_X_y, get_X_y_labelled, get_X_y_unlabelled
    from train import pretrain, train

    FULL_SET = False
    FULL_SET_PROP = 0.3
    NUM_EPOCHS = 2
    NORMALIZE_DATA = True

    UPDATE_RATIO = 0.1

    QUERY_METHOD = 'entropy'
    QUERY_ALPHA = 0.5
    QUERY_K = 100


    train_set, test_set, validate_set = get_data(normalize_data=NORMALIZE_DATA)
    if FULL_SET:
        X, y = get_X_y(train_set, true_labels=True, prop=FULL_SET_PROP)
        X_test, y_test = get_X_y(test_set, true_labels=True, prop=FULL_SET_PROP)
        X_unlabelled, y_unlabelled = None, None
    else:
        X, y = get_X_y_labelled(train_set)
        X_test, y_test = get_X_y_labelled(test_set)

        X_unlabelled, y_unlabelled = get_X_y_unlabelled(train_set)

        X_unlabelled = X_unlabelled.to_numpy().astype(np.float32)
        y_unlabelled = y_unlabelled.to_numpy().astype(np.float32)

    
    # from naive_data import naive_get_data
    # X, y, X_test, y_test = naive_get_data()

    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    # breakpoint()

    model = get_nn()
    # model.fit(X, y)

    # Test pretraining
    # results = pretrain(model, X, y, X_test, y_test)
    # print(results)

    # Test training (involves pretraining as well)
    results = train(model, X, y, X_test, y_test, X_unlabelled=X_unlabelled, y_unlabelled=y_unlabelled, num_epochs=NUM_EPOCHS, update_ratio=UPDATE_RATIO, query_method=QUERY_METHOD, query_alpha=QUERY_ALPHA, query_K=QUERY_K)
    # print(results)
    breakpoint()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
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

    ax[3].plot(model.get_metrics(type='auc', set='training'), label='Training')
    ax[3].plot(model.get_metrics(type='auc', set='validation'), label='Validation')
    ax[3].set_title('AUC')
    ax[3].legend()

    plt.savefig('../plots/nn_metrics.png')

    # breakpoint()
