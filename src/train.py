from datasets import Dataset
import jax
import numpy as np
import xgboost as xgb

# Train model
def train(model, X, y, num_epochs=10, batch_size=512):
    X = X.to_numpy()
    y = y.to_numpy()
    ds = Dataset.from_dict({"X": X, "y": y})
    ds = ds.with_format("jax")  

    # do initial training
    # model.fit(X[:10], y[:10])  # Warmup

    # Begin Training!
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        ds.shuffle(seed=epoch)
        for batch_num, batch in enumerate(ds.iter(batch_size=batch_size)):
            print(f"Batch {batch_num}")
            X_train, y_train = batch['X'], batch['y']
            model.fit(X_train, y_train)

            #if iteration % 10 == 0:
            #     model.fit()
            # else: 
            #     model.update()

# If script is run directly, we will run a training trial
if __name__ == '__main__':
    from xgb import get_xgb
    from naive_data import naive_get_data
    X, y = naive_get_data()
    model = get_xgb()
    train(model, X, y)