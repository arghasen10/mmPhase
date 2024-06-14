import os
import numpy as np
import pandas as pd
import pickle
from scipy import stats as st
from helper import *

data_folder = "datasets"
bin_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

all_files_data = []

for file_name in bin_files:
    print("Processing file ", file_name)
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()
    
    rangeHeatmaps = []
    all_range_index = []
    all_consistent_peaks = []

    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        rangeResultabs = np.abs(rangeResult)
        rangeHeatmap = np.sum(rangeResultabs, axis=(0,1))
        rangeHeatmaps.append(rangeHeatmap)

        intensity_threshold = 100
        peaks_min_intensity_threshold = find_peaks_in_range_data(rangeResult, pointCloudProcessCFG, intensity_threshold)
        threshold = 10
        all_range_index.append(peaks_min_intensity_threshold)
    for frame_no in range(total_frame_number):
        if frame_no < total_frame_number-1:
            current_peaks = all_range_index[frame_no]
            next_peaks = all_range_index[frame_no+1]
            consistent_peaks = get_consistent_peaks(current_peaks, next_peaks, threshold)
            all_consistent_peaks.append(consistent_peaks)
            vel_array_frame = np.array(get_velocity(rangeResult, all_consistent_peaks[frame_no], info_dict)).flatten()
            mean_velocity = (vel_array_frame.mean())
            print(mean_velocity)
            all_files_data.append({'rangeHeatmap': rangeHeatmaps, 'velocity': mean_velocity})

# Convert list of frame data dictionaries to a DataFrame
df = pd.DataFrame(all_files_data)

# Save the merged DataFrame to a .pkl file
pkl_filename = 'merged_data.pkl'
with open(pkl_filename, 'wb') as f:
    pickle.dump(df, f)

print(f"Merged data saved to {pkl_filename}")
