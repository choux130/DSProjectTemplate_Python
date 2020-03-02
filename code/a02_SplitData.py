# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Split Data into Train and Test dataset
# 
# #### Input 
# * winequality-red.csv
# 
# #### Output
# * train.parquet
# * test.parquet

# %%
import os
import pandas as pd 
import pyarrow
from sklearn.model_selection import train_test_split
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default='123')
    parser.add_argument('--percent_train', default='0.8')
    args = parser.parse_args()
    
    random_seed = int(args.random_seed)
    percent_train = float(args.percent_train)

    # file and directory info
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    interim_folder = "/data/interim/"
    raw_folder = "/data/raw/"
    file_name = 'winequality-red.csv'
    var_y = 'quality'

    # %%
    data = pd.read_csv(project_dir + raw_folder + file_name, error_bad_lines = False)
    train, test = train_test_split(data, train_size = percent_train, random_state = random_seed)

    # %%
    train.to_parquet(project_dir + interim_folder + 'train.parquet')
    test.to_parquet(project_dir + interim_folder + 'test.parquet')