from helper import *
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from trajectory_modified import Trajectory
from sklearn.preprocessing import MinMaxScaler
from statistics import mode

fig = plt.figure()
ax = fig.add_subplot(111,)
scat = ax.scatter([], [], s=50)


def update(frame,raw_poincloud_data_for_plot,cluster_labels):
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    # ax.set_zlim(0, 1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    current_data = raw_poincloud_data_for_plot[frame]
    print(current_data.shape)
    labels = cluster_labels[frame]
    unique_labels = np.unique(labels)
    scatters = []
    # for label in unique_labels:
    #     cluster_data = current_data[labels == label]
    scatter = ax.scatter(current_data[:, 0], current_data[:, 1], s=50) #, current_data[:, 2])
    scatter = ax.scatter([np.median(current_data[:, 0]),], [np.median(current_data[:, 1]),], s=50,color='red') #, current_data[:, 2])
    # std_x = np.std(current_data[:, 0])
    # std_y = np.std(current_data[:, 1])
    # std_z = np.std(current_data[:, 2])
    # std_dev_str = f'Stdev X: {std_x:.2f}, Y: {std_y:.2f}, Z: {std_z:.2f}'
    # ax.legend(loc='upper center', ncol=3, fontsize='small')
    ax.set_title(f'3D Scatter Plot Animation (Frame {frame})')
    fig.tight_layout()
    # current_labels = cluster_labels[frame]
    # doppler_shifts = current_data[:,3]
    # normalized_doppler_shifts = (doppler_shifts-doppler_shifts.min())/(doppler_shifts.max()-doppler_shifts.min())
    # scat = ax.scatter(current_data[:, 0], current_data[:, 1], current_data[:, 2],c=current_labels, cmap='viridis', marker='o')
    return scatter,



class DetectStatic:
    def __init__(self,vel_eps=0.0005, xyz_std=1, min_points=50):
        self.min_max_scaler=MinMaxScaler()
        self.vel_scanner = DBSCAN(eps=vel_eps, min_samples=5)
        self.xy_scanner= lambda e: (np.linalg.norm(e[:,:2].std(axis=0))<xyz_std) and (e.shape[0]>min_points)
        
    def static_clusters(self,pointCloud): #col_vec of vel
        self.vel_scanner.fit(self.min_max_scaler.fit_transform(pointCloud[:,[3]]))
        clusters=self.vel_scanner.labels_
        unique_cids=np.unique(clusters)
        #e[0]-->ucid, e[1]--> pointCloud
        return dict(filter(lambda e: self.xy_scanner(e[1]),{ucid:pointCloud[np.where(clusters==ucid)] for ucid in unique_cids}.items()))
    
    
def point_cloud_frames(file_name = None):
    info_dict = get_info(file_name)
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    skipped_frames = 0
    pointCloudProcessCFG = PointCloudProcessCFG()
    raw_poincloud_data_for_plot = []
    cluster_labels = []
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
        velocities.append(np.array(get_velocity(rangeResult, peaks, info_dict)).flatten())
        pointCloud = frame2pointcloud(dopplerResult, pointCloudProcessCFG)
        yield pointCloud
        
gen=point_cloud_frames(file_name = "2024-03-29_vicon_210.bin")
total_data = []
total_ids = []
total_frames=0
sdetect=DetectStatic()
for frame in gen:
    # if total_frames >= 30:
    #     break
    clusters=sdetect.static_clusters(frame)
    datas=[];ids=[]
    for c,p in clusters.items():
        print(c, p.shape)
        datas.extend(p)
        ids.extend([c]*len(p))
    if len(datas) == 0:
        continue
    print(f"Frame number: {total_frames}")
    total_data.append(np.array(datas))
    total_ids.append(np.array(ids))
    total_frames+=1
anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True, fargs=(total_data,total_ids,))
# # anim = FuncAnimation(fig, update, frames=total_frame_number-skipped_frames, interval=50, blit=True, fargs=(raw_poincloud_data_for_plot,))
anim.save('3d_scatter_animation_new.gif', writer='ffmpeg', fps=10)