# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Create Encoder from training data
# * Here is about how we are going to process data based on the information in train dataset. It can be encoders for the text and categorical variables or mean and standard deviation for scaling numeric variables or imputation values for the missing values. 
# * The output from this step will be used to transform data into the format that the model can train or predict on.
# 
# #### Input 
# * train.parquet
# 
# #### Output
# * preprocessor.pkl

# %%
import os 
import pandas as pd
import cloudpickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='mean')
    args = parser.parse_args()
    
    method = str(args.method)

    # file and directory info
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    interim_folder = "/data/interim/"
    processed_folder = "/data/processed/"
    var_y = 'quality'

    train = pd.read_parquet(project_dir + interim_folder + 'train.parquet')
    var_numeric_x = train.columns.tolist()
    var_numeric_x.remove(var_y)

    preprocessor_X = Pipeline(steps=[
        ('x_imputer', SimpleImputer(strategy = method)),
        ('x_scaler', StandardScaler())
    ])

    preprocessor_Y = Pipeline(steps=[
        ('y_scaler', MinMaxScaler())
    ])

    train_X_untransformed = train.drop([var_y], axis=1)
    train_Y_untransformed = train[var_y].to_frame()

    preprocessor_X.fit(train_X_untransformed)
    preprocessor_Y.fit(train_Y_untransformed)

    with open(project_dir + processed_folder + 'preprocessor_X.pkl', 'wb') as f:
        cloudpickle.dump(preprocessor_X, f)
    with open(project_dir + processed_folder + 'preprocessor_Y.pkl', 'wb') as f:
        cloudpickle.dump(preprocessor_Y, f)
