import pandas as pd
import numpy as np
from typing import Optional, Iterable
from sklearn.preprocessing import normalize


def train_test_split(
    df, train_size=0.6, validate_size=0.2, test_size=0.2, seed=20142015
):
    """
    Split the data into training, validation, and test sets and save them to csv files.
    """
    np.random.seed(seed)

    # # Split the data into training validation and test sets with 60%, 20%, 20% respectively
    # train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

    # Split the data into training validation and test sets using the provided sizes
    train, validate, test = np.split(
        df.sample(frac=1),
        [int(train_size * len(df)), int((train_size + validate_size) * len(df))],
    )

    # Save the data to csv files
    train.to_csv("../data/train.csv", index=False)
    validate.to_csv("../data/validate.csv", index=False)
    test.to_csv("../data/test.csv", index=False)


def preprocess_data(df, log_scaling=True, drop_highly_correlated_features=True):
    """
    Preprocess the data by removing any missing values and encoding the categorical variables.
    """
    df = pd.get_dummies(df)

    if log_scaling:
        columns_to_transform = [
            "days_since_request",
            "zip_count_4w",
            "proposed_credit_limit",
        ]

        # Apply natural logarithm transformation to specified columns
        df[columns_to_transform] = np.log1p(df[columns_to_transform])

    if drop_highly_correlated_features:
        cor = df.corr()
        mask = np.triu(np.ones_like(cor))

        def correlation(dataset, threshold):
            col_corr = set()
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j]) > threshold:
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)
            return col_corr

        corr_features = correlation(df, 0.7)

        df = df.drop(corr_features, axis=1)
        # df = df.drop('payment_type_AA', axis=1)

    return df


def _get_sample_numbers(N, alpha, beta, gamma):
    a = beta * gamma * N
    b = beta * (1 - gamma) * N
    c = (alpha - beta * gamma) * N
    d = (1 - alpha - beta + beta * gamma) * N

    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)

    # account for rounding errors
    if a + b + c + d != N:
        diff = N - (a + b + c + d)
        d = d + diff

    assert a + b + c + d == N

    return a, b, c, d


def _splitdfs_by_weights(
    tp_subdf: pd.DataFrame,
    tn_subdf: pd.DataFrame,
    tp_weights: Iterable[float],
    tn_weights: Iterable[float],
    a: int,
    b: int,
    seed = 20142015
):
    np.random.seed(seed)
    # select a labels from true positives
    a_idxs = np.random.choice(tp_subdf.index, a, replace=False, p=tp_weights)
    # split true positives into a and c
    a_subdf: pd.DataFrame = tp_subdf.loc[a_idxs]
    c_subdf: pd.DataFrame = tp_subdf.drop(index=a_idxs)

    # select b labels from true negatives
    b_idxs = np.random.choice(tn_subdf.index, b, replace=False, p=tn_weights)
    # split true negatives into b and d
    b_subdf: pd.DataFrame = tn_subdf.loc[b_idxs]
    d_subdf: pd.DataFrame = tn_subdf.drop(index=b_idxs)

    return a_subdf, b_subdf, c_subdf, d_subdf


def _get_weights(
    probs: Optional[Iterable[float]],
    tp_mask: pd.Series,
    tn_mask: pd.Series,
):
    # split weights into true positives and true negatives
    if probs is not None:
        tp_weights = probs[tp_mask]
        tn_weights = probs[tn_mask]
            
        # normalize the weights
        tp_weights = np.array(tp_weights).reshape(-1, 1)
        tn_weights = np.array(tn_weights).reshape(-1, 1)
        tp_weights = normalize(tp_weights, axis=0, norm="l1").flatten()
        tn_weights = normalize(tn_weights, axis=0, norm="l1").flatten()
    else:
        tp_weights = None
        tn_weights = None

    return tp_weights, tn_weights


def remove_labels(
    df: pd.DataFrame,
    labelled_proportion: float,
    labelled_positive_proportion: float,
    probs: Optional[Iterable[float]] = None,
    top_probs_only: bool = False,
    seed=20142015,
):
    """
    Remove some labels from the dataset (replace with np.nan).

    true_pos_proportion: num true samples/num all samples
    human_labelled_proportion: num human labelled samples/num all samples
    human_labelled_positive_proportion: num human labelled positive samples/num all human labelled samples
    probs: weights for each sample (e.g. from predict_proba)
                - if None
                    all samples are equally likely to be selected
                - otherwise
                    higher weight means higher probability of label being kept
                    must be same length as df
                    weight corresponds to relative probability of keeping label within class,
                    i.e. a positive with weight 0.8 is twice as likely stay labelled as a positive with weight 0.4
                    labelled_proportion and labelled_postive_proportion still enforced
    """

    np.random.seed(seed)

    # check that the proportions are valid
    assert labelled_proportion >= 0 and labelled_proportion <= 1
    assert labelled_positive_proportion >= 0 and labelled_positive_proportion <= 1
    # check that the weights are valid
    if probs is not None:
        assert all([w >= 0 for w in probs])
        assert len(probs) == len(df)
        df["probs"] = probs

    # add a column to the dataframe that is the fraud_bool column but with the labels removed
    df["fraud_masked"] = df["fraud_bool"]

    tp_mask: pd.Series = df["fraud_bool"] == 1
    tn_mask: pd.Series = df["fraud_bool"] == 0

    N = len(df)
    alpha = sum(tp_mask) / N
    beta = labelled_proportion
    gamma = labelled_positive_proportion

    a, b, c, d = _get_sample_numbers(N, alpha, beta, gamma)
    
    # split the dataframe into true positives and true negatives
    tp_subdf: pd.DataFrame = df[tp_mask]
    tn_subdf: pd.DataFrame = df[tn_mask]
    
    if probs is not None and top_probs_only:
        tp_weights = probs[tp_mask]
        tn_weights = probs[tn_mask]
        a_idxs = tp_subdf.nlargest(a, 'probs').index
        a_subdf = tp_subdf.loc[a_idxs]
        b_idxs = tn_subdf.nlargest(b, 'probs').index
        b_subdf = tn_subdf.loc[b_idxs]
        c_subdf = tp_subdf.drop(index=a_idxs)
        d_subdf = tn_subdf.drop(index=b_idxs)
    else:
        # get weights for each sample
        tp_weights, tn_weights = _get_weights(probs, tp_mask, tn_mask)

        # split dfs into labelled true positives, labelled true negatives, unlabelled true positives, unlabelled true negatives
        a_subdf, b_subdf, c_subdf, d_subdf = _splitdfs_by_weights(
            tp_subdf, tn_subdf, tp_weights, tn_weights, a, b
        )

    # remove labels from c_subdf and d_subdf
    c_subdf["fraud_masked"] = np.nan
    d_subdf["fraud_masked"] = np.nan

    # concatenate a_subdf, b_subdf, c_subdf, d_subdf
    df = pd.concat([a_subdf, b_subdf, c_subdf, d_subdf]).sort_index()

    assert len(df) == N
    assert df.index.is_unique

    if probs is not None:
        df = df.drop(columns=["probs"])

    return df

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    df = pd.read_csv("../data/archive/Base.csv")

    # Preprocess the data
    print("Preprocessing features...")
    df = preprocess_data(df, log_scaling=True, drop_highly_correlated_features=True)

    # Remove some labels
    labelled_proportion = 0.01
    labelled_positive_proportion = 0.4

    # If using the weak learner results for de-labelling:
    # import them here, 
    # then change the probs argument to newprobs
    # (and set top_probs_only as desired)
    weak_learner_results = pd.read_csv("./preprocess/weaklearner_weights.csv")
    newprobs = weak_learner_results["y_prob"]

    print("Removing labels...")
    df = remove_labels(
        df,
        labelled_proportion=labelled_proportion,
        labelled_positive_proportion=labelled_positive_proportion,
        probs=None,
        # probs=newprobs,
        top_probs_only=True,
    )

    # Split the data
    print("Splitting data...")
    train_test_split(df)

    print("Data preprocessing done.")
