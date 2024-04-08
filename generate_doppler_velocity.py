import glob
import pandas as pd
import json
import numpy as np
from scipy import stats as st
import pickle
from datetime import datetime
import csv
from tqdm import tqdm
from helper import *


def extract_res(file_name):
    print(file_name)
    info_dict=get_info(file_name.split("/")[-1])
    run_data_read_only_sensor(info_dict)
    bin_filename='datasets/only_sensor'+info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = info_dict[' Nf'][0]
    pointCloudProcessCFG = PointCloudProcessCFG()
    prev_range_bins = []
    max_range_indices = []
    max_doppler_indices = []
    overlapped_range_bins = []
    velocity_array = []
    mode_velocities = []
    mean_velocities = []
    for frame_no in tqdm(range(total_frame_number)):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        max_range_index, range_bins=iterative_range_bins_detection(rangeResult,pointCloudProcessCFG)
        max_range_indices.append(max_range_index)
        if frame_no < 5:
            overlapped_range_bins.append(range_bins)
            prev_range_bins = range_bins
        else:
            last_frame_idx = len(overlapped_range_bins)
            curr_ranges = set()
            for prev_range_bin in prev_range_bins:
                for cur_range_bin in range_bins:
                    if abs(prev_range_bin - cur_range_bin) <= 5:
                        #if within +/- 3, then keep the range bins 
                        curr_ranges.add(cur_range_bin)
            prev_range_bins = range_bins
            overlapped_range_bins.append(np.array(list(curr_ranges)))
            range_bins = overlapped_range_bins[-1]

        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        max_doppler_index, doppler_bins = iterative_doppler_bins_selection(dopplerResult,pointCloudProcessCFG,range_bins, max_range_index)
        max_doppler_indices.append(max_doppler_index)
        doppler_velocity = max_doppler_index*0.0343
        velocity_array.append(doppler_velocity)
    bin_reader.close()  
    call_destructor(info_dict)
    return velocity_array


if __name__ == "__main__":
    erros = []
    ratios = []
    dict_list = []
    files = glob.glob("datasets/*.bin")
    for f in files:
        velocity_array = extract_res(f)
        dict_list.append({'filename': f, 'estimated_speed': velocity_array})
    
    with open('estimated_speed_doppler.json', 'w') as file:
        json.dump(dict_list, file)
    