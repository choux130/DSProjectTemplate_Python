# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Prepare Data for Model 
# 
# * Convert data to the format that the model can consume. This preparation process should be consistent for train, test and prediction dataset. 
# 
# #### Input
# * train.parquet
# * test.parquet
# * preprocessor.pkl
# 
# #### Output
# * train_X.parquet
# * train_Y.parquet
# * test_X.parquet
# * test_Y.parquet

# %%
import os
import pandas as pd
import pyarrow 
import cloudpickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    interim_folder = "/data/interim/"
    processed_folder = "/data/processed/"
    var_y = 'quality'

    train = pd.read_parquet(project_dir + interim_folder + 'train.parquet')
    test = pd.read_parquet(project_dir + interim_folder + 'test.parquet')

    with open(project_dir + processed_folder + 'preprocessor_X.pkl', 'rb') as f:
                preprocessor_X = cloudpickle.load(f)
    with open(project_dir + processed_folder + 'preprocessor_Y.pkl', 'rb') as f:
                preprocessor_Y = cloudpickle.load(f)

    train_X_untransformed = train.drop([var_y], axis=1)
    train_Y_untransformed = train[var_y].to_frame()
    test_X_untransformed = test.drop([var_y], axis=1)
    test_Y_untransformed = test[var_y].to_frame()

    train_X = pd.DataFrame(preprocessor_X.transform(train_X_untransformed), columns = train_X_untransformed.columns)
    train_Y = pd.DataFrame(preprocessor_Y.transform(train_Y_untransformed), columns = train_Y_untransformed.columns)
    test_X = pd.DataFrame(preprocessor_X.transform(test_X_untransformed), columns = test_X_untransformed.columns)
    test_Y = pd.DataFrame(preprocessor_Y.transform(test_Y_untransformed), columns = test_Y_untransformed.columns)

    train_X.to_parquet(project_dir + interim_folder + 'train_X.parquet')
    train_Y.to_parquet(project_dir + interim_folder + 'train_Y.parquet')
    test_X.to_parquet(project_dir + interim_folder + 'test_X.parquet')
    test_Y.to_parquet(project_dir + interim_folder + 'test_Y.parquet')