import os
from os.path import isfile, join
from statistics import mode
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from helper import *

data_folder = "datasets"
bin_files = [f for f in os.listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

fig, ax = plt.subplots()

def process_file(file_name):
    print("Processing file ", file_name)
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()
    print_info(info_dict)

    all_range_index = []
    all_mode_peak = []
    all_consistent_peaks = []
    heatmaps = []

    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        rangeResult = np.abs(rangeResult)
        rangeHeatmap = np.sum(rangeResult, axis=(0,1))
        heatmaps.append(rangeHeatmap)
    
        range_result_absnormal_split = []
        for i in range(pointCloudProcessCFG.frameConfig.numTxAntennas):
            for j in range(pointCloudProcessCFG.frameConfig.numRxAntennas):
                r_r = np.abs(rangeResult[i][j])
                r_r[:,0:10] = 0
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
        intensities_peaks = [[range_abs_combined_nparray_collapsed[idx],idx] for idx in peaks]
        peaks = [i[1] for i in sorted(intensities_peaks, reverse=True)[:3]]
        print("ALL:", peaks)
        all_range_index.append(peaks)
        if frame_no == 0:
            all_consistent_peaks.append(peaks)
            all_mode_peak.append(mode(peaks))
        else:
            previous_peaks = all_range_index[frame_no-1]
            current_peaks = all_range_index[frame_no]
            consistent_peaks = get_consistent_peaks(previous_peaks, current_peaks, threshold=10)
            print("CONSISTENT:", consistent_peaks)
            all_consistent_peaks.append(consistent_peaks)
            all_mode_peak.append(mode(consistent_peaks))

    return heatmaps, all_consistent_peaks, all_mode_peak, file_name

def update(frame):
    ax.clear()

    sns.heatmap(frame[0], ax=ax, cbar=False)
    for peak in frame[1]:
        ax.axvline(x=peak, color='r', linestyle='--')
    ax.axvline(x=frame[2], color='g', linestyle='-', linewidth=4)
    ax.text(0.5, 1.05, frame[3], ha='center', va='center', transform=ax.transAxes, fontsize=12)

for file_name in bin_files:
    frames = []
    heatmaps, consistent_peaks, mode_peak, file_name = process_file(file_name)
    for i, heatmap in enumerate(heatmaps):
        if i < len(consistent_peaks):
            frames.append((heatmap, consistent_peaks[i], mode_peak[i], file_name))
    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    mywriter = animation.PillowWriter(fps=5)
    ani.save("animations/"+file_name+".gif", writer=mywriter)
    print("Animation", file_name, "saved.")
