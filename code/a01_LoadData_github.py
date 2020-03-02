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

if __name__ == "__main__":
    # %%
    csv_url = 'https://raw.githubusercontent.com/choux130/Wine-Quality-Dataset/master/winequality-red.csv'
    data = pd.read_csv(csv_url, error_bad_lines = False)

    # file and directory info
    file_name = 'winequality-red.csv'
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # save the file to the assigned directory
    final_file_dir = os.path.join(project_dir, 'data/raw', file_name)
    data.to_csv(final_file_dir, index=None, header=True, float_format='%g')