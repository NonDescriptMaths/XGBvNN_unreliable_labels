from datasets import Dataset
import jax
import numpy as np
import xgboost as xgb

def pretrain(model, X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray):
    ds = Dataset.from_dict({"X": X, "y": y})
    ds = ds.with_format("jax")  

    # Begin Training!
    ds.shuffle(seed=0)
    X_train, y_train = ds['X'], ds['y']
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    metrics = model.get_metrics()
    return metrics

# Train model
def train(model, X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, num_epochs=10, batch_size='max'):
    ds = Dataset.from_dict({"X": X, "y": y})
    ds = ds.with_format("jax")  

    # Begin Training!
    if batch_size == 'max':
        batch_size = len(X)

    all_metrics = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        ds.shuffle(seed=epoch)

        for batch_num, batch in enumerate(ds.iter(batch_size=batch_size)):
            print(f"Batch {batch_num}")
            X_train, y_train = batch['X'], batch['y']
            
            if (epoch % 10 == 0) and (batch_num == 0):
                pretrain(model, X_train, y_train, X_test, y_test)
                metric = model.get_metrics()
            else: 
                model.update(X_train, y_train, eval_metric="logloss", eval_set=[(X_train, y_train), (X_test, y_test)])
                metric = model.get_metrics()
            
            all_metrics.append(metric)
    return all_metrics

# If script is run directly, we will run a training trial
if __name__ == '__main__':
    from xgb import get_xgb
    from naive_data import naive_get_data
    X, y, X_test, y_test = naive_get_data()
    X = X.to_numpy()
    y = y.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    model = get_xgb()
    # Test pretraining
    results = pretrain(model, X, y, X_test, y_test)
    print(results)
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
    # results  =train(model, X, y, X_test, y_test, num_epochs=3)
    # print(results)