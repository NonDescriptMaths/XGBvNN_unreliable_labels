# LV Unreliable labels project
A githb repo for a two week project unreliable labels in classifiers


## Important Links
These are important epic cool links
+ [Kaggle Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data)
+ [Gradient Boosted Fraud Detection on Data](https://www.kaggle.com/code/dskswu/frauddetectionsystem)
+ [Nyan Cat 100 Hours](https://www.youtube.com/watch?v=9J62hGda9BQ&pp=ygUSbnlhbiBjYXQgMTAwIGhvdXJz) 

# Bookkeeping
 - Setting up data in a directory, `.csv` files are currently in `.gitignore` but we can change if need
    - Download `.zip` file from Kaggle
    - Extract `.zip` to `/data`
    - Directory should look like `data/archive/` `Base.csv`, `Variant I.csv`, `Variant II.csv` etc.
 - Then to preprocess the data and generate train/validation/test sets, `cd` into `src/` and run `python preprocess.py`. (Or use the functions inside `preprocess.py` in other python scripts if you want.)
