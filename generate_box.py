import json
import numpy as np


def extract_filenames_and_data(data, data_key, op):
    filenames_data = {}
    if op == 0:
        for entry in data:
            filenames_data[entry['filename']] = entry[data_key]
    elif op == 1:
        for entry in data:
            filename = '/'.join(entry['filename'].split('/')[1:]).split('.')[0]+'.bin'
            filenames_data[filename] = entry[data_key]
    else:
        for entry in data:
            filename = entry['filename'].split('.')[0]+'.bin'
            filenames_data[filename] = entry[data_key]
    return filenames_data


with open('estimated_speed_doppler.json', 'r') as dop_file, open('estimated_speed_mmPhase.json', 'r') as mmPhase_file, open('gt_speed.json', 'r') as gt_file, open('imuestimate.json', 'r') as imu_file, open('milliEgo/milliegoestimate.json', 'r') as milliego_file:
    gt = json.load(gt_file)
    mmPhase = json.load(mmPhase_file)
    dop = json.load(dop_file)
    imu = json.load(imu_file)
    milliego = json.load(milliego_file)

gt_vals = extract_filenames_and_data(gt, 'gt_speed', 0)
mmphase_vals = extract_filenames_and_data(mmPhase, 'estimated_speed', 0)
dop_vals = extract_filenames_and_data(dop, 'estimated_speed', 0)
imu_vals = extract_filenames_and_data(imu, 'imu_estimate', 0)
imu_vals = extract_filenames_and_data(imu, 'imu_estimate', 1)
milliego_vals = extract_filenames_and_data(milliego, 'estimate', 2)
# print(milliego_vals)

common_filenames = set(gt_vals.keys()) & set(mmphase_vals.keys()) & set(dop_vals.keys()) & imu_vals.keys() & set(milliego_vals.keys())

captured_data = {}
for filename in common_filenames:
    captured_data[filename] = {
        'gt': gt_vals.get(filename),
        'mmphase': mmphase_vals.get(filename),
        'dop': dop_vals.get(filename),
        'imu': imu_vals.get(filename),
        'milliego': milliego_vals.get(filename),
    }

with open('final_result.json', 'w') as file:
    json.dump(captured_data,file)