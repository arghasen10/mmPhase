# check for all files and print which files are inconsistent and how much is the difference between peaks
from os import listdir
from os.path import isfile, join
from helper import *

data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

total_consistent = 0
total_inconsistent = 0

for file_name in bin_files:
    print("Processing file:", file_name)
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()

    prev_x_location = None
    consistent = True
    diff = 0  # Track the maximum difference causing inconsistency

    # Iterate through each frame in the current file
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        rangeResult = np.abs(rangeResult)
        
        # Sum along the axis to get the 2D heatmap
        rangeHeatmap = np.sum(rangeResult, axis=(0, 1))

        max_indices = np.argmax(rangeHeatmap, axis=1)
        top_5_indices = np.sort(max_indices)[-5:]
        x_location = top_5_indices[0]

        # Check consistency with previous frame
        if prev_x_location is not None:
            diff = prev_x_location - x_location
            if diff > 10:
                consistent = False
                break
        
        prev_x_location = x_location

    # Print consistency status for the current file
    if consistent:
        total_consistent += 1
        print("File:", file_name, "is consistent.")
    else:
        total_inconsistent += 1
        print("File:", file_name, "is inconsistent. Difference:", diff)

print("Total Consistent Files:", total_consistent)
print("Total Inconsistent Files:", total_inconsistent)
