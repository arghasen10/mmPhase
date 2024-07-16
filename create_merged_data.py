import os
import pickle
import numpy as np
import pandas as pd
from helper import *
from generate_range_angle_plots import *
import math


def find_initial_l_r(rangeResult, peaks):
    dopplerResult = dopplerFFT(rangeResult, frameConfig)      
    dopplerResultabs = np.absolute(dopplerResult)
    dopplerResultabs = np.sum(dopplerResultabs, axis=(0, 1))
    _, cfar_result = get_coordinates(dopplerResultabs)
    az_angle_map = get_azimuthal_angle(dopplerResult, cfar_result)
    range_angle = np.zeros((256, 64), dtype=np.complex_)

    for key, value in az_angle_map.items():      
        range_angle[key[1]] += np.abs(value)           
    
    range_angle = np.abs(range_angle)
    locate_dots = []

    for peak in peaks:
        locate_dots.append([peak, np.argmax(range_angle[peak])])
    x = [point[0] for point in locate_dots]
    y = [point[1] for point in locate_dots]
    
    r_vals = [x_val * cfg.RANGE_RESOLUTION for x_val in x]
    l_vals = [math.cos((i - 32) * 1.875 / math.pi) * r_val for r_val, i in zip(r_vals, y)]

    return r_vals, l_vals


data_folder = "datasets"
bin_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

velocities = []
rangeResults = []
L_R = []
r_vals, l_vals = 0, 0

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
        rangeResults.append(rangeResult)
        
        range_result_absnormal_split = []
        for i in range(pointCloudProcessCFG.frameConfig.numTxAntennas):
            for j in range(pointCloudProcessCFG.frameConfig.numRxAntennas):
                r_r = np.abs(rangeResult[i][j])
                r_r[:, 0:10] = 0
                min_val = np.min(r_r)
                max_val = np.max(r_r)
                r_r_normalise = (r_r - min_val) / (max_val - min_val) * 1000
                range_result_absnormal_split.append(r_r_normalise)
        range_abs_combined_nparray = np.zeros((pointCloudProcessCFG.frameConfig.numLoopsPerFrame, pointCloudProcessCFG.frameConfig.numADCSamples))
        for ele in range_result_absnormal_split:
            range_abs_combined_nparray += ele
        range_abs_combined_nparray /= (pointCloudProcessCFG.frameConfig.numTxAntennas * pointCloudProcessCFG.frameConfig.numRxAntennas)
        range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / pointCloudProcessCFG.frameConfig.numLoopsPerFrame
        peaks, _ = find_peaks(range_abs_combined_nparray_collapsed)
        intensities_peaks = [[range_abs_combined_nparray_collapsed[idx], idx] for idx in peaks]
        peaks = [i[1] for i in sorted(intensities_peaks, reverse=True)[:3]]
        all_range_index.append(peaks)
        # For first frame take all peaks as consistent peaks
        if frame_no == 0:
            all_consistent_peaks.append(peaks)
            r_vals, l_vals = find_initial_l_r(rangeResult, peaks)
        else:
            previous_peaks = all_range_index[frame_no - 1]
            current_peaks = all_range_index[frame_no]
            consistent_peaks = get_consistent_peaks(previous_peaks, current_peaks, threshold=10)
            all_consistent_peaks.append(consistent_peaks)
        vel_array_frame = np.array(get_velocity(rangeResult, all_consistent_peaks[frame_no], info_dict)).flatten()
        mean_velocity = (vel_array_frame.mean())
        velocities.append(mean_velocity)
        L_R.append([l_vals, r_vals])


# Read velocities from velocity_comparison_results.csv
velocity_comparison_results = pd.read_csv('velocity_comparison_results.csv')
vicon_dict = dict(zip(velocity_comparison_results['file_name'], velocity_comparison_results['estimated_velocity']))

# Convert lists to numpy arrays
rangeResults_array = np.array(rangeResults)
velocities_array = np.array(velocities)
L_R_array = np.array(L_R)

# Collect vicon velocities corresponding to bin files
vicon_velocities = [vicon_dict[file_name] for file_name in bin_files]

# Create the final dictionary
data_dict = {'rangeResult': rangeResults_array, 'velocity': velocities_array * 100, 'L_R': L_R_array, 'vicon': np.array(vicon_velocities)}

# Save the dictionary as a pickle file
with open('merged_data_quant.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Merged data saved to merged_data_quant.pkl")
