# Get consistent peaks for each frame, get velocity for frame from equation, mean and compare with ground truth
import csv
from os import listdir
from os.path import isfile, join
from scipy import stats as st
from helper import *

data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

dict_list = []
velocity_array = []
mode_velocities = []
mean_velocities = []
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
    print_info(info_dict)
    total_frames = total_frames + total_frame_number

    all_range_index = []
    all_consistent_peaks = []
    consistent = True
    # Iterate through each frame in the current file
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        intensity_threshold = 100
        peaks_min_intensity_threshold = find_peaks_in_range_data(rangeResult, pointCloudProcessCFG, intensity_threshold)

        all_range_index.append(peaks_min_intensity_threshold)

        threshold = 10
        if frame_no > 0:
            current_peaks = all_range_index[frame_no-1]
            next_peaks = all_range_index[frame_no]
            if check_consistency_of_frame(current_peaks, next_peaks, threshold):
                consistent_peaks = get_consistent_peaks(current_peaks, next_peaks, threshold)
                all_consistent_peaks.append(consistent_peaks)
            else:
                print("Inconsistent frame found")
                inconsistent_files.append(file_name)
                inconsistent_frames = inconsistent_frames + 1
                consistent = False
    
    if not inconsistent_files:
        for frame_no in range(len(all_consistent_peaks)):
            vel_array_frame = np.array(get_velocity(rangeResult,all_consistent_peaks[frame_no],info_dict)).flatten()
            velocity_array.append(vel_array_frame)
            mode_velocities.append(st.mode(vel_array_frame)[0])
            mean_velocities.append(vel_array_frame.mean())
        estimated_speed = [item for sublist in velocity_array for item in sublist]
        Vb_speed = float(info_dict[' Vb'][0])
        average_estimated_speed = sum(mean_velocities) / len(mean_velocities) if mean_velocities else 0
        speed_difference = average_estimated_speed - Vb_speed
        
        dict_list.append({'filename': file_name, 'estimated_speed': average_estimated_speed, 'Vb_speed': Vb_speed, 'speed_difference': speed_difference})
        
if inconsistent_files:
    # Print the number of inconsistent files and their names
    inconsistent_files = set(inconsistent_files)
    print(f"Number of files with inconsistent data: {len(inconsistent_files)}")
    print("Number of inconsistent frames:", inconsistent_frames, "/", total_frames)
    print("Files with inconsistent data:")
    for file_name in inconsistent_files:
        print(file_name)
else:
    # Save the data to a CSV
    csv_filename = 'estimated_speed_vs_Vb.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'estimated_speed', 'Vb_speed', 'speed_difference']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)