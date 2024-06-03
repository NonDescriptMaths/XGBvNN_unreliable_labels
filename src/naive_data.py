import pandas as pd

def naive_get_data():
    card_df = pd.read_csv('/kaggle/input/bank-account-fraud-dataset-neurips-2022/Base.csv')
    card_org = card_df.copy()
    # Reduce the size of the dataset
    card_df = card_df.sample(n = 400000,random_state=42)
    # One hot encode
    card_df = pd.get_dummies(card_df)
    return card_df