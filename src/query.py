import jax.numpy as jnp


def ground_truth_cross_entropy(model, K, unlabelled_data, labelled_data=None):
    '''
    --- Ground Truth Cross Entropy Sampling ---
    Returns the top K samples w.r.t the cross entropy of the predictions.
    Requires knowledge of true distribution, not practical in real-world scenarios.
    Maybe a good benchmark?
    -------------------------------------------
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    predictions = model.predict(unlabelled_data)
    '''
    predictions: (x,y,y_hat) for 0 <= y_hat <=1
    '''
    # Calculate the cross entropy of the predictions
    p = jnp.clip(predictions[2], 1e-10, 1 - 1e-10)
    q = jnp.clip(predictions[1], 1e-10, 1 - 1e-10)
    ce = -p * jnp.log2(q) - (1 - p) * jnp.log2(1 - q)
    return ce[:K]

def cross_entropy(model, K, unlabelled_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    predictions = model.predict(unlabelled_data)

    ce = predictions.cross_entropy()

    #pick top K samples
    ce.sort()

    return ce[:K]

def margin(model, K, unlabelled_data, labelled_data=None):
    '''
    --- Margin Sampling ---
    Returns the K samples with the smallest margin of the predictions.
    -----------------------
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    predictions = model.predict(unlabelled_data)

    # Calculate the margin of the predictions
    margin = jnp.abs(predictions[2] - 0.5)

    # Sort the margin values in ascending order and get the indices
    indices = jnp.argsort(margin)

    # Return the top K samples
    return unlabelled_data[indices[:K]]

def shannon_entropy(model, K, unlabelled_data, labelled_data=None):
    '''
    --- Shannon Entropy Sampling ---
    Returns the top K samples w.r.t Shannon entropy, produces equvilent outpt to margin
    -------------------------------
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    predictions = model.predict(unlabelled_data)
    '''
    predictions: (x,y,y_hat) for 0 <= y_hat <=1
    '''
    # Calculate the entropy of the predictions
    p = predictions[2]
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    entropy = -p * jnp.log2(p) - (1 - p) * jnp.log2(1 - p)

    # Sort the entropy values in descending order and get the indices
    indices = jnp.argsort(-entropy)

    # Return the top K samples
    return indices[:K]



def random(model, K, unlabelled_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''

    return unlabelled_data.sample(K)