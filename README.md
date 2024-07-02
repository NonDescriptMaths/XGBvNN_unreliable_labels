# LV Unreliable labels project
A githb repo for a two week project unreliable labels in binary classifiers for industry. 


## Important Links
These are important links
+ [Kaggle Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data)
+ [Gradient Boosted Fraud Detection on Data](https://www.kaggle.com/code/dskswu/frauddetectionsystem)
+ [Nyan Cat 100 Hours](https://www.youtube.com/watch?v=9J62hGda9BQ&pp=ygUSbnlhbiBjYXQgMTAwIGhvdXJz)
+ [Lilian's blog - Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/)
+ [Query strategies - `libact` package](https://libact.readthedocs.io/en/latest/overview.html#querystrategy)
+ [How to install `libact`](https://pypi.org/project/libact/)
+ [Query strategies - `scikit-activeml` package](https://scikit-activeml.github.io/scikit-activeml-docs/generated/strategy_overview.html)
+ [Information density query strategy - `modAL` package](https://modal-python.readthedocs.io/en/latest/content/query_strategies/information_density.html)

# Bookkeeping
 - Setting up data in a directory, `.csv` files are currently in `.gitignore` but we can change if need
    - Download `.zip` file from Kaggle
    - Extract `.zip` to `/data`
    - Directory should look like `data/archive/` `Base.csv`, `Variant I.csv`, `Variant II.csv` etc.
 - Then to preprocess the data and generate train/validation/test sets, `cd` into `src/` and run `python preprocess.py`. (Or use the functions inside `preprocess.py` in other python scripts if you want.)
    - This will create the files `train.csv', `validate.csv', and `test.csv' inside the folder `data'.
    - There are two fraud features: `fraud_bool' (fully labelled) and `fraud_masked' (~99% masked with NaNs s.t. 40% of unmasked samples are fraudulent)

