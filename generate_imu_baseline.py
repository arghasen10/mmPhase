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
    imu_df = pd.DataFrame(columns=['time', 'z', 'y', 'x', 'yaw', 'pitch', 'roll', 'speed'])
    ra_df = pd.DataFrame(columns=['time', 'ra', 'speed'])
    time_difference = pd.Timedelta(seconds=6.276132)
    gt_speed = 0
    file_found = False
    with open('/home/argha/github/mmPhase/dataset.csv', 'r') as file:
        data = csv.reader(file)
        for d in data:
            if d[0].split('.')[0] == f.split('.')[0].split("milliEgo/datasets/")[-1]:
                gt_speed = float(d[11])/100
                print("gt_speed from hand ", gt_speed)
    
    if gt_speed == -0.01:
        with open('/home/argha/github/mmPhase/data.json', 'r') as file:
            gt_data = json.load(file)
        for d in gt_data:
            if d['filename'].split('.')[0] == f.split('.')[0].split("milliEgo/")[-1]:
                file_found = True
                gt_speed = d['vicon_gt_final']/100
    if file_found:
        print("gt_speed: ", gt_speed)
    else:
        print('file not found', f.split('.')[0].split("/")[-1])
        return
    with open(f, 'rb') as file:
        loaded_arrays = pickle.load(file)
    imu_data = loaded_arrays[1]
    imu_time = loaded_arrays[3]
    datetime_imu = list(map(convert_to_datetime, imu_time))
    shifted_datetime_imu = [dt - time_difference for dt in datetime_imu]
    for imu, t in zip(imu_data, shifted_datetime_imu):
        imu_df = imu_df.append(pd.Series([t, *imu, gt_speed], index=imu_df.columns), ignore_index=True)
    imu_df['dt'] = imu_df['time'].diff().dt.total_seconds()
    imu_df.loc[0,'dt'] = 0
    imu_df['vx'] = imu_df['x'].cumsum() * np.mean(imu_df['dt'].values)
    imu_df = imu_df.dropna()
    # imu_df['vy'] = imu_df['y'].cumsum() * imu_df['dt']
    # imu_df['absolute_velocity'] = (imu_df['vx']**2 + imu_df['vy']**2)**0.5
    
    data_without_nan = imu_df['vx'].values[~np.isnan(imu_df['vx'].values)]
    # print("Mode velocity: ", st.mode(data_without_nan))
    print("Mean velocty: ", data_without_nan.mean())
    return np.abs(gt_speed-data_without_nan.mean())
    



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
        data_dict = {'filename':f, 'error': error_val}
        erros.append(error_val)
        dict_list.append(data_dict)

    print("Mean: ", np.array(erros).mean())
    print("Mode: ", st.mode(erros))
    with open('imuerror.json', 'w') as file:
        json.dump(dict_list, file)
    print(erros)
    print(sum(erros)/len(erros))