import pandas as pd
import numpy as np

def train_test_split(df, train_size=0.6, validate_size=0.2, test_size=0.2, seed=20142015):
    '''
    Split the data into training, validation, and test sets and save them to csv files.
    '''
    np.random.seed(seed)

    # # Split the data into training validation and test sets with 60%, 20%, 20% respectively
    # train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

    # Split the data into training validation and test sets using the provided sizes
    train, validate, test = np.split(df.sample(frac=1), [int(train_size*len(df)), int((train_size+validate_size)*len(df))])

    # Save the data to csv files
    train.to_csv('../data/train.csv', index=False)
    validate.to_csv('../data/validate.csv', index=False)
    test.to_csv('../data/test.csv', index=False)


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('../data/archive/Base.csv')

    # Split the data
    train_test_split(df)

    print('Data split into training, validation, and test sets.')