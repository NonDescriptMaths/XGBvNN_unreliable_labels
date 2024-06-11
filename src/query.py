import jax as jax
import jax.numpy as jnp
from skactiveml.pool import UncertaintySampling
from sklearn.preprocessing import StandardScaler

'''
For these sampling and similarity functions, we will assume that we are given a 
prepredicted dataset (predicted_data), which consists of:

predicted_data[0:-2] = covariates
predicted_data[-1] = predicted prob of labels via the model

'''

# Defining Similarity functions 
###################################
def representativeness_rbf(unlabelled_data, sigma, beta):
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
        similarities = jnp.exp(euclidean_dist / sigma)
        # Calculate representativeness of sample i
        representativeness = representativeness.at[i].set(jnp.mean(similarities)**beta)
    
    return representativeness

def representativeness_re(unlabelled_data, beta):
    '''
    
    '''
    
    scaler = StandardScaler()
    unlabelled_data = scaler.fit_transform(unlabelled_data)
    
    n_samples = unlabelled_data.shape[0]
    # Initialise vector
    representativeness = jnp.zeros(n_samples)
    
    for i in range(n_samples):
        euclidean_dist = jnp.linalg.norm(unlabelled_data - unlabelled_data[i], axis=1)
        # Calculate euclidean similarity
        similarities = euclidean_dist
        # Calculate representativeness of sample i
        representativeness = representativeness.at[i].set(jnp.mean(similarities)**beta)
    return representativeness

###################################

###################################
# Define Uncertainty samplers (some of which use similarity functions)
###################################
def centropy(K, predicted_data, labelled_data=None):
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
    p = predicted_data[:,-1]
    p = jnp.clip(predictions[2], 1e-10, 1 - 1e-10)
    q_value = jnp.mean(predictions[1])
    q = jnp.full(len(p),q_value)
    ce = -q * jnp.log2(p) - (1 - q) * jnp.log2(1 - p)
    indices = jnp.argsort(-ce)
    return indices[:K]

def margin(K, predicted_data, labelled_data=None):
    '''
    --- Margin Sampling ---
    Returns the K samples with the smallest margin of the predictions.
    -----------------------
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    p = predicted_data[:,-1]
    # Calculate the margin of the predictions
    margin = jnp.abs(p - 0.5)

    # Sort the margin values in ascending order and get the indices
    indices = jnp.argsort(margin)

    # Return the top K samples
    return indices[:K]

def entropy(K, predicted_data, labelled_data=None):
    '''
    --- Shannon Entropy Sampling ---
    Returns the top K samples w.r.t Shannon entropy, produces equvilent outpt to margin
    -------------------------------
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    
    # Calculate the entropy of the predictions
    p = predicted_data[:,-1]
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    entropy = -p * jnp.log2(p) - (1 - p) * jnp.log2(1 - p)

    # Sort the entropy values in descending order and get the indices
    indices = jnp.argsort(-entropy)

    # Return the top K samples
    return indices[:K]

def random(K, predicted_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    key = jax.random.PRNGKey(0)  
    return jax.random.choice(key, len(predicted_data), K, replace=False)

def entrepRE(K,predicted_data,labelled_data = None,repres_data = []):
    '''
    Entropy with similarity constraints
    inputs: 
            K: number of queries to select
            predicted_data: prediction data from model
            labelled_data: currently labelled data
    outputs: indices of selected samples
    '''
    p = predicted_data[:,-1]
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    entropy = -p * jnp.log2(p) - (1 - p) * jnp.log2(1 - p)
    loss_func = entropy * repres_data
    indices = jnp.argsort(-loss_func)
    return indices[:K]

def entrepRBF(K,predicted_data,labelled_data=None,repres_data = []):
    '''
    Entropy with similarity constraints
    inputs: model: xgboost or nn
            K: number of queries to select
            unlabelled_data: currently unlabelled data
            labelled_data: currently labelled data
    outputs: indices of selected samples
    '''
    p = predicted_data[:,-1]
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    entropy = -p * jnp.log2(p) - (1 - p) * jnp.log2(1 - p)
    loss_func = entropy * repres_data
    indices = jnp.argsort(-loss_func)
    return indices[:K]

###################################
# Define main query function 
###################################

def sampler(predicted_data,
            K= 30,
            alpha = 0.5,
            threshold_for_fraud = 0.8,
            method = 'entropy',
            RE_beta = 1,
            RBF_sigma = 0.5,
            RBF_beta = 1,
            debug = False):
    '''
    --- Sampler Function ---
    ------------------------
    input:
    K: Size of sample
    alpha: proportion of sample required to be FRAUD cases
    threshold_for_fraud: probability at which fraud is decided
    method: method for finding informative points, can be 'entropy','centropy','random','margin','entrepRE','entrepRBF'
            'entropy': Shannon Entropy Sampling
            'centropy': Ground Truth Cross Entropy Sampling NOT RECOMMENDED
            'random': Random Sampling
            'margin': Margin Sampling
            'entrepRE': Entropy with similarity constraints via euclidean norm
            'entrepRBF': Entropy with similarity constraints via RBF kernel
    '''
    # generate a certain amount of fraud cases
    if debug:
        print("...Sampling...")
        print("Fraud/Sample ratio:", alpha)
        print("Method: ", method)
    n_fraud = int(K*alpha)
    n_non_fraud = K - n_fraud
    # get the indices of fraud cases
    fraud_indices = jnp.where(predicted_data[:,-1] > threshold_for_fraud)[0]
    fraud_indices_sample = fraud_indices[:n_fraud]
    # get the indices of non-fraud cases for the sample
    non_fraud_indices = jnp.where(predicted_data[:,-1] <= threshold_for_fraud)
    non_fraud_data = predicted_data[non_fraud_indices]

    #Generate similarity data
    if method == 'entrepRE':
        repres_data = representativeness_re(predicted_data[:,0:-1], beta=RE_beta)
    elif method == 'entrepRBF':
        repres_data = representativeness_rbf(predicted_data[:,0:-1], sigma=RBF_sigma, bRBF_beta=1)
    

    if method == 'entropy':
        indices = entropy(n_non_fraud,predicted_data)
    elif method == 'centropy':
        indices = centropy(n_non_fraud,predicted_data)
    elif method == 'random':
        indices = random(n_non_fraud,predicted_data)
    elif method == 'margin':
        indices = margin(n_non_fraud,predicted_data)
    elif method == 'entrepRE':
        indices = entrepRE(n_non_fraud,predicted_data,repres_data=repres_data)
    elif method == 'entrepRBF':
        indices = entrepRBF(n_non_fraud,predicted_data,repres_data=repres_data)
    else:
        raise ValueError('Method not supported by sampler')
    # print the indices
    if debug:
        print(f"Fraud indices: \n {fraud_indices_sample}")
        print(f"Uncertainty indices: \n {indices}")

    # combine and output
    return jnp.concatenate([fraud_indices_sample,indices],axis = 0)



#Compute the representativeness of the unlabelled data for RBF, euclidean distance and beta
'''


predicted_data = jnp.array([[1, 2, 3, 4, 0.5],
                            [5, 6, 7, 8, 0.2],
                            [9, 10, 11, 12, 0.7],
                            [13, 14, 15, 16, 0.9]])
'''

#repres_RBF = representativeness_rbf(predicted_data[:,1:-2], sigma=0.5, beta=1)
#repres_RE = representativeness_reciprocal_euclidean(predicted_data[:,1:-2], beta=1)

