from helper import *

file_name = "datasets/2024-03-29_vicon_145.bin"

info_dict=get_info(file_name.split("/")[-1])
run_data_read_only_sensor(info_dict)
bin_filename='datasets/only_sensor'+info_dict['filename'][0]
bin_reader = RawDataReader(bin_filename)
total_frame_number = info_dict[' Nf'][0]
pointCloudProcessCFG = PointCloudProcessCFG()
for frame_no in range(total_frame_number):
    bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
    np_frame = bin2np_frame(bin_frame)
    frameConfig = pointCloudProcessCFG.frameConfig
    reshapedFrame = frameReshape(np_frame, frameConfig)
    rangeResult = rangeFFT(reshapedFrame, frameConfig)
    print(rangeResult.shape) # No of tx antenna x no of rx antenna x no of chirps x no of adc samples
    rangeResult = np.abs(rangeResult)
    print(rangeResult.shape)
    rangeHeatmap = np.sum(rangeResult, axis=(0,1))
    sns.heatmap(rangeHeatmap)
    # plt.savefig('heatmap_test.png')
    plt.show()

