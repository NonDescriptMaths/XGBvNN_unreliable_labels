from datasets import Dataset
import jax


# Train model
def train(model, data):
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