# check consistency for one file only
from helper import *

file_name = "datasets/2024-03-29_vicon_15.bin"

info_dict = get_info(file_name.split("/")[-1])
run_data_read_only_sensor(info_dict)
bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
bin_reader = RawDataReader(bin_filename)
total_frame_number = int(info_dict[' Nf'][0])
pointCloudProcessCFG = PointCloudProcessCFG()

prev_x_location = None
consistent = True

for frame_no in range(total_frame_number):
    bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
    np_frame = bin2np_frame(bin_frame)
    frameConfig = pointCloudProcessCFG.frameConfig
    reshapedFrame = frameReshape(np_frame, frameConfig)
    rangeResult = rangeFFT(reshapedFrame, frameConfig)
    rangeResult = np.abs(rangeResult)
    
    # Sum along the axis to get the 2D heatmap
    rangeHeatmap = np.sum(rangeResult, axis=(0, 1))

    sns.heatmap(rangeHeatmap)
    plt.show()

    max_indices = np.argmax(rangeHeatmap, axis=1)
    top_5_indices = np.sort(max_indices)[-5:]
    x_location = top_5_indices[0]

    print("Frame:", frame_no, "X Location:", x_location)

    if prev_x_location is not None:
        if prev_x_location - x_location > 2:
            consistent = False
            break
    
    prev_x_location = x_location

# Check consistency
if consistent:
    print("Consistent: The x location of the maximum vertical line decreases consistently by less than or equal to 2 units across consecutive frames.")
else:
    print("Inconsistent: The x location of the maximum vertical line does not decrease consistently by less than or equal to 2 units across consecutive frames.")
