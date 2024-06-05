import jax.numpy as jnp
from skactiveml.pool import UncertaintySampling

def cross_entropy(model, K, unlabelled_data, labelled_data=None):
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
    ASSUMPTIONS
    predictions: (x,y,y_hat) for 0 <= y_hat <=1
    '''
    # Calculate the cross entropy of the predictions
    p = jnp.clip(predictions[2], 1e-10, 1 - 1e-10)
    q_value = jnp.mean(predictions[1])
    q = jnp.full(len(p),q_value)
    ce = -q * jnp.log2(p) - (1 - q) * jnp.log2(1 - p)
    indices = jnp.argsort(-ce)
    return indices[:K]

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



def representativeness_rbf(unlabelled_data, sigma):
    """
    Computes the representativeness for each sample in the unlabelled data.

    Parameters:
    - unlabelled_data: The dataset (as a 2D array) of unlabelled samples.
    - sigma: The scaling parameter for the rbf.

    Returns:
    - representativeness: A 1D array containing the representativeness score of each sample.
    """
    
    n_samples = unlabelled_data.shape[0]
    # Initialise vector
    representativeness = jnp.zeros(n_samples)
    
    for i in range(n_samples):
        euclidean_dist = jnp.linalg.norm(unlabelled_data - unlabelled_data[i], axis=1)
        # Calculate rbf
        similarities = jnp.exp(-euclidean_dist / sigma)
        # Calculate representativeness of sample i
        representativeness[i] = jnp.mean(similarities)
    
    return representativeness


def representativeness_reciprocal_euclidean(unlabelled_data):
    
    n_samples = unlabelled_data.shape[0]
    # Initialise vector
    representativeness = jnp.zeros(n_samples)
    
    for i in range(n_samples):
        euclidean_dist = jnp.linalg.norm(unlabelled_data - unlabelled_data[i], axis=1)
        # Calculate euclidean similarity
        similarities = 1 / euclidean_dist
        # Calculate representativeness of sample i
        representativeness[i] = jnp.mean(similarities)
    
    return representativeness



def random(model, K, unlabelled_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''

    return unlabelled_data.sample(K)
