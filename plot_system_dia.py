from helper import *
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.cluster.hierarchy as hcluster
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from trajectory_modified import Trajectory


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter([], [], [],s=50)

    
def calculate_centroids(data, labels):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append((label, centroid))
    return centroids


def track_clusters(previous_centroids, current_centroids, distance_threshold=0.1):
    tracked_clusters = {}
    for current_label, current_centroid in current_centroids:
        for prev_label, prev_centroid in previous_centroids:
            if euclidean(prev_centroid, current_centroid) < distance_threshold:
                tracked_clusters[current_label] = prev_label
                break
        else:
            tracked_clusters[current_label] = current_label  # New cluster
    return tracked_clusters


def calculate_combined_std(point_cloud_data):
    std_x = np.std(point_cloud_data[:, 0])
    std_y = np.std(point_cloud_data[:, 1])
    combined_std = np.sqrt(std_x**2 + std_y**2)
    return combined_std


def track_static_clusters(filtered_data, file_name, output_folder='clustered_scatter_plots', ):
    raw_poincloud_data = []
    for frame_no, data in enumerate(filtered_data):
        X = data[:, [0, 1,]]  # x, y
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(X)
        cluster_labels = clustering.labels_
        if frame_no == 0:
            previous_centroids = calculate_centroids(data, cluster_labels)
            continue
        
        current_centroids = calculate_centroids(data, cluster_labels)
        tracked_clusters = track_clusters(previous_centroids, current_centroids, distance_threshold=0.5)
        tracked_labels = np.array([tracked_clusters[label] for label in cluster_labels])

        raw_poincloud_data.append(data)
        return raw_poincloud_data, tracked_labels
    
def get_tracked_cluster_info(cluster_pcds, cluster_labels):
    centroids_dicts = []
    for cluster_pcd, cluster_label in zip(cluster_pcds, cluster_labels):
        unique_labels = np.unique(cluster_label)
        centroids_dict = {}
        for label in unique_labels:
            cluster_points = cluster_pcd[cluster_label == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids_dict[label] = centroid
            centroids_dicts.append(centroids_dict)
    return centroids_dicts

def apply_clustering_and_plot(filtered_data, file_name, output_folder='clustered_scatter_plots', ):
    global fig
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    raw_poincloud_data_for_plot = []
    all_cluster_labels = []
    save_individual_figures = False
    for frame_no, data in enumerate(filtered_data):
        X = data[:, [0, 1,]]  # x, y
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(X)
        cluster_labels = clustering.labels_
        if frame_no == 0:
            previous_centroids = calculate_centroids(data, cluster_labels)
            continue
        
        current_centroids = calculate_centroids(data, cluster_labels)
        tracked_clusters = track_clusters(previous_centroids, current_centroids, distance_threshold=0.5)
        tracked_labels = np.array([tracked_clusters[label] for label in cluster_labels])

        raw_poincloud_data_for_plot.append(data)
        all_cluster_labels.append(tracked_labels)
    
    # if not save_individual_figures:
    #     anim = FuncAnimation(fig, update, frames=len(raw_poincloud_data_for_plot), interval=50, blit=True, fargs=(raw_poincloud_data_for_plot,all_cluster_labels,))
    #     gif_name = file_name+"_tracked_static_3d_scatter_animation.gif"
    #     anim.save(gif_name, writer='ffmpeg', fps=10)
    # else:
    #     for frame_no, data in enumerate(raw_poincloud_data_for_plot):
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.clear()
    #         ax.set_xlim(0, 1)
    #         ax.set_ylim(0, 1)
    #         ax.set_zlim(0, 1)
    #         ax.set_xlabel('X axis')
    #         ax.set_ylabel('Y axis')
    #         ax.set_zlabel('Z axis')
    #         current_data = data
    #         labels = all_cluster_labels[frame_no]
    #         unique_labels = np.unique(labels)
    #         if len(Counter(labels)) == 1:
    #             continue
    #         for label in unique_labels:
    #             cluster_data = current_data[labels == label]
    #             if len(cluster_data) >= 10:
    #                 ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {label}')

    #         std_x = np.std(current_data[:, 0])
    #         std_y = np.std(current_data[:, 1])
    #         std_z = np.std(current_data[:, 2])
    #         std_dev_str = f'Stdev X: {std_x:.2f}, Y: {std_y:.2f}, Z: {std_z:.2f}'
    #         ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='small')
    #         ax.set_title(f'3D Scatter Plot (Frame {frame_no})\n{std_dev_str}')
    #         fig.tight_layout()

    #         # Save the plot as a PNG file
    #         file_name = os.path.join(output_folder, f'frame_{frame_no:03d}.png')
    #         plt.savefig(file_name)
    #         plt.close(fig)
    return raw_poincloud_data_for_plot, all_cluster_labels

def save_scatter_plots(raw_poincloud_data_for_plot, cluster_labels, output_folder='scatter_plots'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for frame_no, (current_data, current_labels) in enumerate(zip(raw_poincloud_data_for_plot, cluster_labels)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        std_x = np.std(current_data[:, 0])
        std_y = np.std(current_data[:, 1])
        combined_std = np.sqrt(std_x**2 + std_y**2)
        cluster_count = len(current_data)
        doppler_shifts = current_data[:, 3]
        velocity = np.mean(doppler_shifts) if len(doppler_shifts) > 0 else 0
        std_dev_str = f'Combined Stdev: {combined_std:.2f}, Velocity: {velocity:.2f}, Count: {cluster_count}'
        
        scat = ax.scatter(current_data[:, 0], current_data[:, 1], current_data[:, 2], c=current_labels, cmap='viridis', marker='o')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(f'Frame {frame_no}\n{std_dev_str}')
        
        file_name = os.path.join(output_folder, f'frame_{frame_no:03d}.png')
        plt.savefig(file_name)
        plt.close(fig)
        
        
def update(frame,raw_poincloud_data_for_plot,):
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 15)
    # ax.set_zlim(0, 5)
    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)
    # ax.set_zlabel('Z', fontsize=28)
    current_data = raw_poincloud_data_for_plot[frame]
    # labels = cluster_labels[frame]
    # unique_labels = np.unique(labels)
    scatters = []
    # for label in unique_labels:
    #     # if label == -1:
    #     #     continue  # Skip noise
    #     cluster_data = current_data[labels == label]
    scatter = ax.scatter(current_data[:, 0], current_data[:, 1], c=current_data[:,3], s=50)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_zticks([])
    plt.grid()
    if frame == 15:
        plt.savefig('image_frame15.png')
    # scatter = ax.scatter(np.median(current_data[:, 0]), np.median(current_data[:, 1]), color='red', s=50)
        # scatters.append(scatter)
    # std_x = np.std(current_data[:, 0])
    # std_y = np.std(current_data[:, 1])
    # std_dev_str = f'Stdev X: {std_x:.2f}, Y: {std_y:.2f}'
    # ax.legend(loc='upper center', ncol=3, fontsize='small')
    ax.set_title(f'2D Scatter Plot Animation (Frame {frame})')
    fig.tight_layout()
    # current_labels = cluster_labels[frame]
    # doppler_shifts = current_data[:,3]
    # normalized_doppler_shifts = (doppler_shifts-doppler_shifts.min())/(doppler_shifts.max()-doppler_shifts.min())
    # scat = ax.scatter(current_data[:, 0], current_data[:, 1], current_data[:, 2],c=current_labels, cmap='viridis', marker='o')
    return scat,


if __name__ == "__main__":
    data_folder = "datasets"
    bin_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]
    for file_name in bin_files:
        file_name = "2024-03-29_vicon_test_14.bin"
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
            if len(pointCloud) == 6:
                skipped_frames+=1
                print("skipped frame: ", frame_no)
                continue
            
            if frame_no == 25:
                ax.clear() 
                ax.set_xlim(0, 2)
                ax.set_ylim(0, 2)
                ax.set_zlim(0, 3)
                ax.set_xlabel('X', fontsize=60)
                ax.set_ylabel('Y', fontsize=60)
                ax.set_zlabel('Z', fontsize=60)
                scatters = []
                scatter = ax.scatter(pointCloud[:, 0], pointCloud[:, 1], pointCloud[:, 2], s=50)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                # plt.grid()
                fig.tight_layout()
                plt.savefig('raw_points.png')
            doppler_shifts = pointCloud[:, 3]
            normalized_doppler_shifts = (doppler_shifts - doppler_shifts.min()) / (doppler_shifts.max() - doppler_shifts.min())
            power_profile = pointCloud[:, 4]
            normalized_power_profile = (power_profile - power_profile.min()) / (power_profile.max() - power_profile.min())

            pointCloud_data = np.concatenate([normalized_doppler_shifts.reshape(-1,1)], axis=1)
            clustering = DBSCAN(eps=0.001, min_samples=10).fit(pointCloud_data)
            clusters=clustering.labels_
            selected_clusters=[]
            # Interesting observation, our cluster selection always selects the cluster with cluster id -1.
            for k,v in Counter(clusters).items():
                cluster_points = pointCloud[clusters == k]
                combined_std = calculate_combined_std(cluster_points)
                if combined_std < 2 and len(cluster_points) > 50:
                    selected_clusters.append(k)

            for selected_cluster in selected_clusters:
                filtered_data = np.array([pointCloud[i] for i, cid in enumerate(clusters) if cid == selected_cluster])
                # raw_poincloud_data_for_plot.append(filtered_data)
                cluster_labels.append([cid for cid in clusters if cid == selected_cluster])
            if frame_no == 25:
                ax.clear() 
                ax.set_xlim(0, 2)
                ax.set_ylim(0, 2)
                ax.set_zlim(0, 3)
                ax.set_xlabel('X', fontsize=60)
                ax.set_ylabel('Y', fontsize=60)
                ax.set_zlabel('Z', fontsize=60)
                scatters = []
                scatter = ax.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], s=50)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                # plt.grid()
                fig.tight_layout()
                plt.savefig('static_clusters_plot.png')
                plt.close()
                unique_labels = np.unique(cluster_labels[-1])
                print(unique_labels)
                colors = sns.color_palette()
                for i, label in enumerate(unique_labels):
                    cluster_data = filtered_data[cluster_labels[-1] == label]
                    ax.clear() 
                    ax.set_xlim(0, 2)
                    ax.set_ylim(0, 2)
                    ax.set_zlim(0, 3)
                    ax.set_xlabel('X', fontsize=50)
                    ax.set_ylabel('Y', fontsize=50)
                    ax.set_zlabel('Z', fontsize=50)
                    scatters = []
                    scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[i], s=50)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                plt.grid()
                fig.tight_layout()
                plt.savefig('clusters_tracked_plot.png')
                plt.close()
        bin_reader.close()
        
        # anim = FuncAnimation(fig, update, frames=total_frame_number-skipped_frames, interval=50, blit=True, fargs=(raw_poincloud_data_for_plot,cluster_labels,))
        anim = FuncAnimation(fig, update, frames=total_frame_number-skipped_frames, interval=50, blit=True, fargs=(raw_poincloud_data_for_plot,))
        # anim.save('2d_scatter_animation_multi.gif', writer='ffmpeg', fps=10)
        anim_filename = f'animation_{file_name}'.split('.')[0]+'.gif'
        print(anim_filename)
        anim.save(anim_filename, writer='ffmpeg', fps=10)
        # save_scatter_plots(raw_poincloud_data_for_plot, cluster_labels)
        # cluster_pcds, cluster_labels = apply_clustering_and_plot(raw_poincloud_data_for_plot, file_name)
        # centroids_dicts = get_tracked_cluster_info(cluster_pcds, cluster_labels)
        # print(centroids_dicts)
        # static_objects = []
        break
