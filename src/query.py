def cross_entropy(model, K, unlabelled_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''
    predictions = model.predict(unlabelled_data)

    ce = predictions.cross_extropy()

    #pick top K samples
    ce.sort()

    return ce[:K]


def random(model, K, unlabelled_data, labelled_data=None):
    '''
    model: prediction model
    K: number of queries to select
    unlabelled_data: currently unlabelled data
    labelled_data: currently labelled data (may not be used)
    '''

    return unlabelled_data.sample(K)