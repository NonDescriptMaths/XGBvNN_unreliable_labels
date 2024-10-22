import pandas as pd
from clean_data import normalize
import os

def get_data(normalize_data=False, harry=False):
    harry_str = '/harry' if harry else ''
    train = pd.read_csv(f'../data{harry_str}/train.csv')
    test = pd.read_csv(f'../data{harry_str}/test.csv')
    validate = pd.read_csv(f'../data{harry_str}/validate.csv')
    
    if normalize_data:
        train = normalize(train)
        test = normalize(test)
        validate = normalize(validate)

    return train, test, validate

def get_X_y(df, true_labels=False, prop=0.5):
    y = df['fraud_bool' if true_labels else 'fraud_masked']
    X = df.drop(['fraud_bool', 'fraud_masked'], axis=1)

    if prop < 1:
        X = X.sample(frac=prop)
        y = y.loc[X.index]
        
    return X, y

def get_X_y_labelled(df):
    df = df.dropna(subset=['fraud_masked'])

    y = df['fraud_masked']
    X = df.drop(['fraud_bool', 'fraud_masked'], axis=1)
    return X, y

def get_X_y_unlabelled(df):
    df = df[df['fraud_masked'].isna()]

    # uncommenting this would return the NaNs, but we want the TRUE labels
    # of the 'unlabelled' data (and we'll just promise not to use them unless
    # they get selected by a query)
    # y = df['fraud_masked']

    y = df['fraud_bool']
    X = df.drop(['fraud_bool', 'fraud_masked'], axis=1)

    return X, y

def check_preprocessed():
    if not os.path.exists('../data/train.csv'):
        raise FileNotFoundError("Data not found. Please run preprocess.py to preprocess the data.")
    elif not os.path.exists('../data/test.csv'):
        raise FileNotFoundError("Data not found. Please run preprocess.py to preprocess the data.")
    elif not os.path.exists('../data/validate.csv'):
        raise FileNotFoundError("Data not found. Please run preprocess.py to preprocess the data.")
    else:
        return True