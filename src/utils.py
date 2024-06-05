import pandas as pd

def get_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    validate = pd.read_csv('../data/validate.csv')
    
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

    y = df['fraud_masked']
    X = df.drop(['fraud_bool', 'fraud_masked'], axis=1)
    return X, y