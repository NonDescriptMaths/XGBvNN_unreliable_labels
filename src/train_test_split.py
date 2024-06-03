import pandas as pd
import numpy as np

np.random.seed(20142015)

# Load the data
df = pd.read_csv('../data/archive/Base.csv')

# Split the data into training validation and test sets with 60%, 20%, 20% respectively
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# Save the data to csv files
train.to_csv('../data/train.csv', index=False)
validate.to_csv('../data/validate.csv', index=False)
test.to_csv('../data/test.csv', index=False)

