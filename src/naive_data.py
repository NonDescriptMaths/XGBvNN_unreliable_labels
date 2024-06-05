import pandas as pd

def naive_get_data(drop_portion_of_labels=0.9, reduce_size=0.2):
    card_df = pd.read_csv('./data/Base.csv')
    # Reduce the size of the dataset
    card_df = card_df.sample(n = 400000,random_state=42)
    # One hot encode
    card_df = pd.get_dummies(card_df)
    # Reduce the size of the dataset
    if reduce_size < 1:
        card_df = card_df.sample(frac=reduce_size, random_state=42)
    # Drop labels
    if drop_portion_of_labels > 0:
        card_df = card_df.drop(card_df[card_df['fraud_bool'] == 1].sample(frac=drop_portion_of_labels).index)
        card_df = card_df.drop(card_df[card_df['fraud_bool'] == 0].sample(frac=drop_portion_of_labels).index)
    # Split data into features and labels
    y = card_df['fraud_bool']
    X = card_df.drop('fraud_bool', axis=1)
    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X, y, X_test, y_test


# If script is run directly, we will run a test of the data fetcher
if __name__ == '__main__':
    X, y = naive_get_data()

    # Print the column names for the features
    print("Column headings:\n", X.columns)
    # Print the first few rows of the features
    print("First few rows:\n", X.head())

    # Print number of labels with fraud_bool == 1 and 0
    print("Counts for fraudulent", y.value_counts())