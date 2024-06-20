import os
import pickle
import numpy as np
from helper import *

data_folder = "datasets"
bin_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

velocities = []
rangeResults = []
L_R = []

for file_name in bin_files:
    print("Processing file ", file_name)
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()
    
    all_range_index = []
    all_consistent_peaks = []

    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        intensity_threshold = 100
        peaks_min_intensity_threshold = find_peaks_in_range_data(rangeResult, pointCloudProcessCFG, intensity_threshold)
        threshold = 10
        all_range_index.append(peaks_min_intensity_threshold)
    
    bin_reader = RawDataReader(bin_filename)
    for frame_no in range(total_frame_number):
        if frame_no < total_frame_number-1:
            bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
            np_frame = bin2np_frame(bin_frame)
            frameConfig = pointCloudProcessCFG.frameConfig
            reshapedFrame = frameReshape(np_frame, frameConfig)
            rangeResult = rangeFFT(reshapedFrame, frameConfig)
            rangeResults.append(rangeHeatmap)
            rangeResultabs = np.abs(rangeResult)
            rangeHeatmap = np.sum(rangeResultabs, axis=(0,1))
            current_peaks = all_range_index[frame_no]
            next_peaks = all_range_index[frame_no+1]
            consistent_peaks = get_consistent_peaks(current_peaks, next_peaks, threshold)
            all_consistent_peaks.append(consistent_peaks)
            vel_array_frame = np.array(get_velocity(rangeResult,all_consistent_peaks[frame_no],info_dict)).flatten()
            mean_velocity = (vel_array_frame.mean())
            velocities.append(mean_velocity)
            L_R.append([info_dict[' L'], info_dict[' R']])
# Convert lists to numpy arrays
rangeResults_array = np.array(rangeResults)
velocities_array = np.array(velocities)
L_R_array = np.array(L_R)

data_dict = {'rangeResult': rangeResults_array, 'velocity': velocities_array, 'L_R': L_R_array}

with open('merged_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Merged data saved to merged_data.pkl")
