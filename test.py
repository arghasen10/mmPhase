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