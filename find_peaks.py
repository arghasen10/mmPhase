from helper import *
from os import listdir
from os.path import isfile, join

data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

inconsistent_files = []
total_frames = 0
inconsistent_frames = 0

for file_name in bin_files:
    print("Processing file ", file_name)
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()

    total_frames = total_frames + total_frame_number

    max_range_index = []
    all_range_index = []
    all_consistent_peaks = []

    # Iterate through each frame in the current file
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        # rangeResult = np.abs(rangeResult)
        intensity_threshold = 200
        peaks_min_intensity_threshold = find_peaks_in_range_data(rangeResult, pointCloudProcessCFG, intensity_threshold)

        # velocity = get_velocity(rangeResult, peaks_min_intensity_threshold, info_dict)
        # # v = np.array(velocity)
        # final_vel = np.mean(np.array([np.mean(v) for v in velocity]))
        # # print(v.shape)
        # print("Velocity: ", final_vel)


        # Store the maximum peak and all detected peaks
        max_range_index.append(np.argmax(np.sum(rangeResult, axis=0) / frameConfig.numLoopsPerFrame))
        all_range_index.append(peaks_min_intensity_threshold)

        threshold = 10
        if frame_no > 0:
            current_peaks = all_range_index[frame_no-1]
            next_peaks = all_range_index[frame_no]
            print("Current:", current_peaks)
            print("Next:", next_peaks)
            if check_consistency_of_frame(current_peaks, next_peaks, threshold):
                consistent_peaks = get_consistent_peaks(current_peaks, next_peaks, threshold)
                print("Consistent:", consistent_peaks)
                all_consistent_peaks.append(consistent_peaks)
            else:
                print("Inconsistent frame found")
                inconsistent_files.append(file_name)
                inconsistent_frames = inconsistent_frames + 1


# Print the number of inconsistent files and their names
inconsistent_files = set(inconsistent_files)
print(f"Number of files with inconsistent data: {len(inconsistent_files)}")
print("Number of inconsitent frames:", inconsistent_frames, "/", total_frames)
if inconsistent_files:
    print("Files with inconsistent data:")
    for file_name in inconsistent_files:
        print(file_name)
