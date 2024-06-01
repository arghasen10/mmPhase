from os import listdir
from os.path import isfile, join
from helper import *

# Define data folder and list of binary files
data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

# Function to check consistency of peaks between frames
def check_consistency_of_frames(all_range_index, threshold):
    for i in range(len(all_range_index) - 1):
        current_peaks = all_range_index[i]
        next_peaks = all_range_index[i + 1]
        # print("Current:", current_peaks)
        # print("Next:", next_peaks)
        if not (any(abs(c - n) <= threshold for n in next_peaks) for c in current_peaks):
            return False
    return True


# List to store names of files with inconsistent data
inconsistent_files = []
# Process each file
for file_name in bin_files:
    print("Processing file ", file_name)
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

    # Check the consistency of frames
    if not check_consistency_of_frames(all_range_index, 2):
        inconsistent_files.append(file_name)
        print("Inconsistent")
    else:
        print("Consistent")

# Print the number of inconsistent files and their names
print(f"Number of files with inconsistent data: {len(inconsistent_files)}")
if inconsistent_files:
    print("Files with inconsistent data:")
    for file_name in inconsistent_files:
        print(file_name)
