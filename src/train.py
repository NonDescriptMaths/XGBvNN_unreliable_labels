from datasets import Dataset
import jax
import numpy as np
import xgboost as xgb
from utils import get_data, get_X_y, get_X_y_labelled
from query import sampler
import jax.numpy as jnp

def pretrain(model, X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray):
    ds = Dataset.from_dict({"X": X, "y": y})
    ds = ds.with_format("jax")  

    # Begin Training!
    ds.shuffle(seed=0)
    X_train, y_train = ds['X'], ds['y']
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    
    return model.get_metrics()

# Train model
def train(model, X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, X_unlabelled:np.ndarray = None, y_unlabelled:np.ndarray = None, num_epochs=20, full_train_every=10, update_ratio=0.1, batch_size='max', query_method='', query_alpha=0.5, query_K=10, query_args={}):
    if query_args == '':
        query_args = {}
    # Separate data into labelled and unlabelled

    # label_idx = np.where(y != -1)[0]

    # breakpoint()

    # ds = Dataset.from_dict({"X": X, "y": y})
    # ds = ds.with_format("jax")

    if query_method != '':
        assert X_unlabelled is not None
        assert y_unlabelled is not None

        # ds_unlabelled = Dataset.from_dict({"X": X_unlabelled, "y": y_unlabelled})

    next_X = X
    next_y = y

    # Begin Training!
    if batch_size == 'max':
        batch_size = len(next_X)

    # all_metrics = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        
        # ds.shuffle(seed=epoch)
        
        #shuffle X and y
        perm = np.random.permutation(len(next_X))
        next_X = next_X[perm]
        next_y = next_y[perm]

        # for batch_num, batch in enumerate(ds.iter(batch_size=batch_size)):
        for batch_num in range(0, len(next_X), batch_size):
            print(f"Batch {batch_num}")
            
            batch = {"X": next_X[batch_num:batch_num+batch_size], "y": next_y[batch_num:batch_num+batch_size]}
            X_train, y_train = batch['X'], batch['y']
            
            if (epoch % full_train_every == 0) and (batch_num == 0) and (full_train_every != -1):
                pretrain(model, X_train, y_train, X_test, y_test)
                # metric = model.get_metrics()
            else: 
                model.update(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
                # metric = model.get_metrics()
            
            # all_metrics.append(metric)

        # select some unlabelled data to label
        if query_method != '':
            # predict on unlabelled data by batching
            predicted = []
            # for batch_num, batch in enumerate(ds_unlabelled.iter(batch_size=10*batch_size)):
            for batch_num in range(0, len(X_unlabelled), 10*batch_size):
                # print(f"Predicting on batch {batch_num//(10*batch_size)}")
                batch = {"X": X_unlabelled[batch_num:batch_num+10*batch_size]}

                if model.__class__.__name__ == "XGBWrapper":
                    logits = model.model.predict_proba(batch['X'])
                else:
                    logits = jax.nn.sigmoid(model.predict(batch['X']))
                
                predicted.append(logits)

            predicted = jnp.concatenate(predicted)
            
            query_idx = sampler(predicted, method=query_method, K=query_K, alpha=query_alpha, **query_args)
            # breakpoint() 
            newly_labelled_X = X_unlabelled[query_idx]
            newly_labelled_y = y_unlabelled[query_idx]

            prev_X = X
            prev_y = y
            
            # update the labelled data for the next iteration
            idx = np.random.randint(prev_X.shape[0], size=int(X_train.shape[0]*(1-update_ratio)))

            next_X = np.concatenate((newly_labelled_X, prev_X[idx, :]), axis=0)
            next_y = np.concatenate((newly_labelled_y, prev_y[idx]), axis=0)

            # update total collection of labelled data
            X = np.concatenate([X, X_unlabelled[query_idx]])
            y = np.concatenate([y, y_unlabelled[query_idx]])
            
            # remove newly labelled data from unlabelled data
            X_unlabelled = np.delete(X_unlabelled, query_idx, axis=0)
            y_unlabelled = np.delete(y_unlabelled, query_idx, axis=0)

            # print(f"Labelled size: {len(X)}, Unlabelled size: {len(X_unlabelled)}")


            # X_labelled = np.concatenate([ds['X'], ds_unlabelled['X'][query_idx]])
            # y_labelled = np.concatenate([ds['y'], ds_unlabelled['y'][query_idx]])
        
            # ds = Dataset.from_dict({"X": X_labelled, "y": y_labelled})
            # ds = ds.with_format("jax")
            # ds_unlabelled = Dataset.from_dict({"X": np.delete(X, query_idx, axis=0), "y": np.delete(y, query_idx, axis=0)})

            # breakpoint()

    return model.get_metrics(log_final=False)

# If script is run directly, we will run a training trial
if __name__ == '__main__':
    from xgb import get_xgb
    train_data, test_data, validate_data = get_data()
    X_train, y_train = get_X_y_labelled(train_data)
    X_val, y_val = get_X_y_labelled(validate_data)
    X_test, y_test = get_X_y_labelled(test_data)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    model = get_xgb(n_estimators=103)
    # Test pretraining
    results = pretrain(model, X_train, y_train, X_test, y_test)
    print(results)

    breakpoint()
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # Plot loss
    import matplotlib.pyplot as plt
    plt.plot(results['loss']['training'], label='train')
    plt.plot(results['loss']['validation'], label='validation')
    plt.legend()
    plt.savefig('plots/loss.png')
    plt.close()
    # Plot AUC
    plt.plot(results['auc']['training'], label='train')
    plt.plot(results['auc']['validation'], label='validation')
    plt.legend()
    plt.savefig('plots/auc.png')
    plt.close()
    # Plot accuracy
    plt.plot(results['acc']['training'], label='train')
    plt.plot(results['acc']['validation'], label='validation')
    plt.legend()
    plt.savefig('plots/acc.png')
    plt.close()

    # Test training
    results = train(model, X_train, y_train, X_test, y_test, num_epochs=20)
    