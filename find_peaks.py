from os import listdir
from os.path import isfile, join
from helper import *

# Define data folder and list of binary files
data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

# Function to normalize and find peaks in range data
def find_peaks_in_range_data(rangeResult, pointcloud_processcfg):
    range_result_absnormal_split = []
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            r_r = np.abs(rangeResult[i][j])
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * 1000
            range_result_absnormal_split.append(r_r_normalise)
    
    # Combine and average the normalized range data
    range_abs_combined_nparray = np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame, pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray += ele
    range_abs_combined_nparray /= (pointcloud_processcfg.frameConfig.numTxAntennas * pointcloud_processcfg.frameConfig.numRxAntennas)
    
    # Collapse range data and find peaks
    range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / pointcloud_processcfg.frameConfig.numLoopsPerFrame
    peaks, _ = find_peaks(range_abs_combined_nparray_collapsed)
    
    # Filter peaks by intensity threshold
    peaks_min_intensity_threshold = []
    for indices in peaks:
        if range_abs_combined_nparray_collapsed[indices] > 150:
            peaks_min_intensity_threshold.append(indices)
    
    return peaks_min_intensity_threshold

# Process each file
for file_name in bin_files:
    print("Processing file:", file_name)
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()

    max_range_index = []
    all_range_index = []

    # Iterate through each frame in the current file
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        rangeResult = np.abs(rangeResult)

        # Find peaks in the current frame's range data
        peaks_min_intensity_threshold = find_peaks_in_range_data(rangeResult, pointCloudProcessCFG)
        
        # Store the maximum peak and all detected peaks
        max_range_index.append(np.argmax(np.sum(rangeResult, axis=0) / frameConfig.numLoopsPerFrame))
        all_range_index.append(peaks_min_intensity_threshold)
        
        # Print or store the results for each frame
        print(f"Frame {frame_no}: Peaks detected at indices {peaks_min_intensity_threshold}")

        rangeHeatmap = np.sum(rangeResult, axis=(0, 1))
        sns.heatmap(rangeHeatmap)
        plt.show()

# At this point, max_range_index and all_range_index will contain the peak information for all frames


