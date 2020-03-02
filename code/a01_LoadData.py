# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Load Data from URL
# 
# #### Output
# * winequality-red.csv

# %%
import os
import pandas as pd
from azure.storage.blob import BlockBlobService

if __name__ == "__main__":
    # azure credentials
    # make sure the credentials are saved as environment variables.
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    # print(connect_str)

    # file and directory info
    container_name = 'datascience-data' 
    file_name = 'winequality-red.csv'
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    final_file_dir = os.path.join(project_dir, 'data/raw', file_name)

    # for azure.storage.blob 2.1.0
    blob = BlockBlobService(connection_string=connect_str)
    blob.get_blob_to_path(container_name, 
                        file_name, 
                        final_file_dir)