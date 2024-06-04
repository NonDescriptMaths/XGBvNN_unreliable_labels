import jax.numpy as jnp


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

def shannon_entropy(model, K, unlabelled_data, labelled_data=None):
    '''
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
    return unlabelled_data[indices[:K]]




def random(model, K, unlabelled_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''

    return unlabelled_data.sample(K)