from helper import *
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

def get_traj(P1, P2, v_b, t, prev_point):
    C1 = (v_b * t) - P1[1] + (P1[0]**2 / (v_b * t - P1[1]))
    B1 = P2[0] + (P2[1] * P1[0] / (v_b * t - P1[1]))
    translation_magnitude = v_b*t #eclid(P1,P2)#v_b*t
    angle = np.arcsin((-B1) / C1)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    prev_point=np.array(prev_point).reshape(-1,1)
    rotated_point = np.dot(rotation_matrix, prev_point)
    translation_vector = translation_magnitude * np.array([[np.sin(angle)],
                                                           [np.cos(angle)]])
    current_point = rotated_point + translation_vector
    return tuple(current_point.flatten())


def eclid(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5


def point_cloud_frames(file_name = None):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()
    velocities = []
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        
        range_result_absnormal_split = []
        for i in range(pointCloudProcessCFG.frameConfig.numTxAntennas):
            for j in range(pointCloudProcessCFG.frameConfig.numRxAntennas):
                r_r = np.abs(rangeResult[i][j])
                r_r[:, 0:10] = 0
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
        intensities_peaks = [[range_abs_combined_nparray_collapsed[idx], idx] for idx in peaks]
        peaks = [i[1] for i in sorted(intensities_peaks, reverse=True)[:3]]
    
        dopplerResult = dopplerFFT(rangeResult, frameConfig)
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG, peaks)
        vel_array_frame = np.array(get_velocity(rangeResult, peaks, info_dict)).flatten()
        mean_velocity = np.median(vel_array_frame)
        yield pointCloud, mean_velocity
        
files = glob.glob('datasets/stick_2024-09-01_*.bin')
for traj_file in files:
    traj_file = traj_file.split('/')[-1]
    plt.cla()
    gen=point_cloud_frames(file_name = traj_file)
    total_data = []
    total_ids = []
    total_frames=0
    trackcluster=defaultdict(lambda :[])
    prev_pointclouds = None
    prev_traj_point = (0,0)
    final_trajectory = [prev_traj_point]
    prev_frame_time = 0
    for frame_no, (points, v_b) in enumerate(gen):
        traj1 = []
        found = False
        frame_time = (frame_no+1)*0.2
        if prev_pointclouds is None:
            prev_pointclouds = points
            continue
        for point1 in points:
            for point2 in prev_pointclouds:
                distance = eclid(point1, point2)
                if distance < 0.1:
                    found = True
                    new_traj_point = get_traj(point1, point2,v_b, 0.2, prev_traj_point)
                    traj1.append(new_traj_point)
        if len(traj1) > 1:
            final_traj_point = (np.median([point[0] for point in traj1]), np.median([point[1] for point in traj1]))
        else:
            print("Skipped frame_no: ", frame_no)
        # print("final_traj_point: ", final_traj_point)
        prev_traj_point = final_traj_point
        final_trajectory.append(prev_traj_point)
        prev_pointclouds = points
        
        
    plt.plot([e[0] for e in final_trajectory],[e[1] for e in final_trajectory])
    save_fig = f"trajectory_plot_{traj_file.split('.')[0]}.png"
    plt.savefig(save_fig)
    plt.close()