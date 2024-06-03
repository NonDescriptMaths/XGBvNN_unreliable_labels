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


def preprocess_data(df, log_scaling=True, drop_highly_correlated_features=True):
    '''
    Preprocess the data by removing any missing values and encoding the categorical variables.
    '''
    df = pd.get_dummies(df)

    if log_scaling:
        columns_to_transform = ['days_since_request', 'zip_count_4w', 'proposed_credit_limit']

        # Apply natural logarithm transformation to specified columns
        df[columns_to_transform] = np.log1p(df[columns_to_transform])


    if drop_highly_correlated_features:
        cor = df.corr()
        mask = np.triu(np.ones_like(cor))

        def correlation(dataset, threshold):
            col_corr = set ()
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if  (corr_matrix.iloc[i, j]) > threshold:
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)
            return col_corr
        corr_features = correlation(df, 0.7)

        df = df.drop(corr_features, axis=1)
        # df = df.drop('payment_type_AA', axis=1)

    return df


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('../data/archive/Base.csv')

    # Preprocess the data
    df = preprocess_data(df, log_scaling=True, drop_highly_correlated_features=True)

    # Split the data
    train_test_split(df)

    print('Data preprocessing done.')