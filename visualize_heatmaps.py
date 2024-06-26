import os
from os.path import isfile, join
from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib.pyplot as plt
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
    all_consistent_peaks = []
    heatmaps = []

    # Iterate through each frame in the current file
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        rangeResult = np.abs(rangeResult)
        rangeHeatmap = np.sum(rangeResult, axis=(0,1))

        # Store heatmap for later animation
        heatmaps.append(rangeHeatmap)
        
        intensity_threshold = 100
        peaks_min_intensity_threshold = find_peaks_in_range_data(rangeResult, pointCloudProcessCFG, intensity_threshold)

        all_range_index.append(peaks_min_intensity_threshold)
        threshold = 10
        if frame_no == 0:
            all_consistent_peaks.append(find_peaks_in_range_data(rangeResult, pointCloudProcessCFG, intensity_threshold))
        else:
            current_peaks = all_consistent_peaks[frame_no-1]
            next_peaks = all_range_index[frame_no]
            consistent_peaks = get_consistent_peaks(current_peaks, next_peaks, threshold)
            all_consistent_peaks.append(consistent_peaks)

    return heatmaps, all_consistent_peaks, file_name

def update(frame):
    ax.clear()
    sns.heatmap(frame[0], ax=ax, cbar=False)
    for peak in frame[1]:
        ax.axvline(x=peak, color='r', linestyle='--')
    ax.text(0.5, 1.05, frame[2], ha='center', va='center', transform=ax.transAxes, fontsize=12)

frames = []
for file_name in bin_files:
    heatmaps, consistent_peaks, file_name = process_file(file_name)
    for i, heatmap in enumerate(heatmaps):
        if i < len(consistent_peaks):
            frames.append((heatmap, consistent_peaks[i], file_name))

ani = FuncAnimation(fig, update, frames=frames, repeat=False)
plt.show()
