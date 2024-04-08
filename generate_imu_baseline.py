import glob
import pandas as pd
import json
import numpy as np
from scipy import stats as st
import pickle
from datetime import datetime
import csv
convert_to_datetime = lambda timestamp: datetime.utcfromtimestamp(timestamp[0])



def collect_imu_speed(f):
    print("filename: ", f)
    imu_df = pd.DataFrame(columns=['time', 'z', 'y', 'x', 'yaw', 'pitch', 'roll'])
    time_difference = pd.Timedelta(seconds=6.276132)
    with open(f, 'rb') as file:
        loaded_arrays = pickle.load(file)
    imu_data = loaded_arrays[1]
    imu_time = loaded_arrays[3]
    datetime_imu = list(map(convert_to_datetime, imu_time))
    shifted_datetime_imu = [dt - time_difference for dt in datetime_imu]
    for imu, t in zip(imu_data, shifted_datetime_imu):
        imu_df = imu_df.append(pd.Series([t, *imu], index=imu_df.columns), ignore_index=True)
    imu_df['dt'] = imu_df['time'].diff().dt.total_seconds()
    imu_df.loc[0,'dt'] = 0
    imu_df['vx'] = imu_df['x'].cumsum() * np.mean(imu_df['dt'].values)
    imu_df = imu_df.dropna()
    data_without_nan = imu_df['vx'].values[~np.isnan(imu_df['vx'].values)]
    print("Mean velocty: ", data_without_nan.mean())
    return np.abs(data_without_nan.mean())
    



if __name__ == "__main__":
    erros = []
    ratios = []
    dict_list = []
    files = glob.glob("milliEgo/datasets/*.pickle")
    files.remove("milliEgo/datasets/2024-03-29_vicon_135.pickle")
    files.remove("milliEgo/datasets/2024-03-29_vicon_210.pickle")
    for f in files:
        error_val = collect_imu_speed(f)
        if error_val is None:
            print("none file found")
            continue
        data_dict = {'filename':f, 'imu_estimate': error_val}
        print(data_dict)
        erros.append(error_val)
        dict_list.append(data_dict)

    with open('imuestimate.json', 'w') as file:
        json.dump(dict_list, file)
