import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from helper import *

data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]


def get_df(file_path):
    raw_data = np.fromfile(file_path, dtype=np.float32)
    df = pd.DataFrame(raw_data)
    return df
dfs = []
for file_name in bin_files:
    file_path = join(data_folder, file_name)
    df = get_df(file_path)
    # Add a column to identify the file it came from
    df['file_name'] = file_name
    dfs.append(df)
# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_data.csv', index=False)