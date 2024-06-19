import struct
import pickle
import numpy as np
import configuration as cfg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import argparse
import pandas as pd
import subprocess
import statistics
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
mode_velocities = []
def read8byte(x):
    return struct.unpack('<hhhh', x)


class FrameConfig:  #
    def __init__(self):
        self.numTxAntennas = cfg.NUM_TX
        self.numRxAntennas = cfg.NUM_RX
        self.numLoopsPerFrame = cfg.LOOPS_PER_FRAME
        self.numADCSamples = cfg.ADC_SAMPLES
        self.numAngleBins = cfg.NUM_ANGLE_BINS

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame
        self.chirpSize = self.numRxAntennas * self.numADCSamples
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame


class PointCloudProcessCFG:  #
    def __init__(self):
        self.frameConfig = FrameConfig()
        self.enableStaticClutterRemoval = False
        self.EnergyTop128 = True
        self.RangeCut = False
        self.outputVelocity = True
        self.outputSNR = True
        self.outputRange = True
        self.outputInMeter = True
        self.EnergyThrMed = True
        self.ConstNoPCD = False
        self.dopplerToLog = False

        dim = 3
        if self.outputVelocity:
            self.velocityDim = dim
            dim += 1
        if self.outputSNR:
            self.SNRDim = dim
            dim += 1
        if self.outputRange:
            self.rangeDim = dim
            dim += 1
        self.couplingSignatureBinFrontIdx = 5
        self.couplingSignatureBinRearIdx = 4
        self.sumCouplingSignatureArray = np.zeros((self.frameConfig.numTxAntennas, self.frameConfig.numRxAntennas,
                                                   self.couplingSignatureBinFrontIdx + self.couplingSignatureBinRearIdx),
                                                  dtype=np.complex128)


class RawDataReader:
    def __init__(self, path):
        self.path = path
        self.ADCBinFile = open(path, 'rb')

    def getNextFrame(self, frameconfig):
        frame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize * 4), dtype=np.int16)
        return frame

    def close(self):
        self.ADCBinFile.close()


def bin2np_frame(bin_frame):  #
    np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=np.complex_)
    np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
    np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
    return np_frame


def frameReshape(frame, frameConfig):  #
    frameWithChirp = np.reshape(frame, (
                                frameConfig.numLoopsPerFrame, frameConfig.numTxAntennas, frameConfig.numRxAntennas, -1))
    return frameWithChirp.transpose(1, 2, 0, 3)


def rangeFFT(reshapedFrame, frameConfig):  #
    windowedBins1D = reshapedFrame
    rangeFFTResult = np.fft.fft(windowedBins1D)
    return rangeFFTResult


def clutter_removal(input_val, axis=0):  #
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)


def dopplerFFT(rangeResult, frameConfig):  #
    windowedBins2D = rangeResult * np.reshape(np.hamming(frameConfig.numLoopsPerFrame), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):  #
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi

    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    elevation_ant_padded[:num_rx, :] = elevation_ant

    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector


def frame2pointcloud(dopplerResult, pointCloudProcessCFG):
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
    if pointCloudProcessCFG.dopplerToLog:
        dopplerResultInDB = np.log10(np.absolute(dopplerResultSumAllAntenna))
    else:
        dopplerResultInDB = np.absolute(dopplerResultSumAllAntenna)

    if pointCloudProcessCFG.RangeCut:  
        dopplerResultInDB[:, :25] = -100
        dopplerResultInDB[:, 125:] = -100
    cfarResult = np.zeros(dopplerResultInDB.shape, bool)
    if pointCloudProcessCFG.EnergyTop128:
        top_size = 128
        energyThre128 = np.partition(dopplerResultInDB.ravel(), 128 * 256 - top_size - 1)[128 * 256 - top_size - 1]
        cfarResult[dopplerResultInDB > energyThre128] = True
    det_peaks_indices = np.argwhere(cfarResult == True)
    R = det_peaks_indices[:, 1].astype(np.float64)
    V = (det_peaks_indices[:, 0] - FrameConfig().numDopplerBins // 2).astype(np.float64)
    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    energy = dopplerResultInDB[cfarResult == True]
    AOAInput = dopplerResult[:, :, cfarResult == True]
    AOAInput = AOAInput.reshape(12, -1)
    if AOAInput.shape[1] == 0:
        return np.array([]).reshape(6, 0)
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)
    x, y, z = x_vec * R, y_vec * R, z_vec * R
    pointCloud = np.concatenate((x, y, z, V, energy, R))
    pointCloud = np.reshape(pointCloud, (6, -1))
    pointCloud = pointCloud[:, y_vec != 0]
    pointCloud = np.transpose(pointCloud, (1, 0))
    if pointCloudProcessCFG.EnergyThrMed:
        idx = np.argwhere(pointCloud[:, 4] > np.median(pointCloud[:, 4])).flatten()
        pointCloud = pointCloud[idx]
    if pointCloudProcessCFG.ConstNoPCD:
        pointCloud = reg_data(pointCloud, 128)  

    return pointCloud


def reg_data(data, pc_size):  #
    pc_tmp = np.zeros((pc_size, 6), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_list = np.random.choice(pc_no, size=pc_size, replace=False)
        pc_tmp = data[pc_list]
    return pc_tmp


def phase_unwrapping(phase_len,phase_cur_frame):
    i=1
    new_signal_phase = phase_cur_frame
    for k,ele in enumerate(new_signal_phase):
        if k==len(new_signal_phase)-1:
            continue
        if new_signal_phase[k+1] - new_signal_phase[k] > 1.5*np.pi:
            new_signal_phase[k+1:] = new_signal_phase[k+1:] - 2*np.pi*np.ones(len(new_signal_phase[k+1:]))
    return np.array(new_signal_phase)


def get_args():
    parser=argparse.ArgumentParser(description="Run the phase_generation script")
    parser.add_argument('-f','--file_name',help="Get the .bin file to process")
    args=parser.parse_args()
    return args


def get_info(args):
    dataset=pd.read_csv('dataset.csv')
    file_name=args
    filtered_row=dataset[dataset['filename']==file_name]
    info_dict={}
    for col in dataset.columns:
        info_dict[col]=filtered_row[col].values
    if len(info_dict['filename'])==0:
        print('Oops! File not found in database. Cross check the file name')
    return info_dict


def print_info(info_dict):
    print('***************************************************************')
    print('Printing the file profile')
    print(f'--filename: {"only_sensor"+info_dict["filename"][0]}')
    print(f'--Length(L in cm): {info_dict[" L"][0]}')
    print(f'--Radial_Length(R in cm): {info_dict[" R"][0]}')
    print(f'--PWM Value: {info_dict[" PWM"][0]}')
    print(f'--A brief desciption: {info_dict[" Description"][0]}')
    print('***************************************************************')


def custom_color_map():
    colors = ["#6495ED", "yellow"]  # Start with blue, end with yellow
    n_bins = 100  # Increase this for smoother transitions
    cmap_name = "customBlueYellow"
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return custom_cmap


def iterative_range_bins_detection(rangeResult,pointcloud_processcfg):
    if pointcloud_processcfg.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult, axis=2)
    range_result_absnormal_split=[]
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            r_r=np.abs(rangeResult[i][j])
            #first 10 range bins i.e 40 cm make it zero
            r_r[:,0:10]=0
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * (1000 - 0) + 0
            range_result_absnormal_split.append(r_r_normalise)
    
    range_abs_combined_nparray=np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame,pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray+=ele
    range_abs_combined_nparray/=(pointcloud_processcfg.frameConfig.numTxAntennas*pointcloud_processcfg.frameConfig.numRxAntennas)
    
    range_abs_combined_nparray_collapsed=np.sum(range_abs_combined_nparray,axis=0)/pointcloud_processcfg.frameConfig.numLoopsPerFrame
    peaks_min_intensity_threshold = np.argsort(range_abs_combined_nparray_collapsed)[::-1][:5]
    max_range_index=np.argmax(range_abs_combined_nparray_collapsed)
    return max_range_index, peaks_min_intensity_threshold


def iterative_doppler_bins_selection(dopplerResult,pointcloud_processcfg,range_peaks, max_range_index):
    doppler_result_absnormal_split=[]
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            d_d=np.abs(dopplerResult[i][j])
            d_d[:,0:10]=0
            min_val = np.min(d_d)
            max_val = np.max(d_d)
            d_d_normalise = (d_d - min_val) / (max_val - min_val) * (1000 - 0) + 0
            doppler_result_absnormal_split.append(d_d_normalise)
    
    doppler_abs_combined_nparray=np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame,pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in doppler_result_absnormal_split:
        doppler_abs_combined_nparray+=ele
    doppler_abs_combined_nparray/=(pointcloud_processcfg.frameConfig.numTxAntennas*pointcloud_processcfg.frameConfig.numRxAntennas)
    
    vel_idx=[]
    for peak in range_peaks:
        vel_idx.append(np.argmax(doppler_abs_combined_nparray[:,peak])-91)
    max_doppler_index = np.argmax(doppler_abs_combined_nparray[:,max_range_index])-91
    return max_doppler_index, vel_idx


def get_phase(r,i):
    if r==0:
        if i>0:
            phase=np.pi/2
        else :
            phase=3*np.pi/2
    elif r>0:
        if i>=0:
            phase=np.arctan(i/r)
        if i<0:
            phase=2*np.pi - np.arctan(-i/r)
    elif r<0:
        if i>=0:
            phase=np.pi - np.arctan(-i/r)
        else:
            phase=np.pi + np.arctan(i/r)
    return phase


def solve_equation(phase_cur_frame,info_dict):
    phase_diff=[]
    for soham in range (1,len(phase_cur_frame)):
        phase_diff.append(phase_cur_frame[soham]-phase_cur_frame[soham-1])
    Tp=cfg.Tp
    Tc=cfg.Tc
    L = info_dict[0]/100
    r0 = info_dict[1]/100
    #L=info_dict[' L'][0]/100
    #r0=info_dict[' R'][0]/100
    roots_of_frame=[]
    for i,val in enumerate(phase_diff):
        c=(phase_diff[i]*0.001/3.14)/(3*(Tp+Tc))
        t=3*(i+1)*(Tp+Tc)
        c1=t*t
        c2=-2*L*t
        c3=L*L-c*c*t*t
        c4=2*L*c*c*t
        c5=-r0*r0*c*c
        coefficients=[c1, c2, c3, c4, c5]
        root=min(np.abs(np.roots(coefficients)))
        roots_of_frame.append(root)
    median_root=np.median(roots_of_frame)
    final_roots=[]
    for root in roots_of_frame:
        if root >0.9*median_root and root<1.1*median_root:
            final_roots.append(root)
        print(root)
    return np.mean(final_roots)


def plot_dopppler_mobicom(doppler_vel_frame_wise,mobicom_vel_frame_wise,info_dict):
    print(doppler_vel_frame_wise)
    print(mobicom_vel_frame_wise)
    for i,ele in enumerate(doppler_vel_frame_wise):
        doppler_vel_frame_wise[i]=doppler_vel_frame_wise[i]*-1
    plt.figure(figsize=(10, 6))
    # plt.plot(doppler_vel_frame_wise, label='Doppler Velocity', marker='o', markersize=5, linestyle='-', linewidth=1, alpha=0.7)
    plt.plot(mobicom_vel_frame_wise, label='MobiCom Velocity', marker='x', linestyle='--', linewidth=1, alpha=0.7)
    plt.plot(mode_velocities, label="Mode Mobicom velocity", marker="*", linestyle="-.", linewidth=1, alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Velocity')
    plt.title(f'Velocity Frame Wise Comparison {info_dict["filename"][0]}\n pwm value={info_dict[" PWM"][0]} \n Expected_speed: {info_dict[" Vb"][0]/100} (the red line)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(y=info_dict[" Vb"][0]/100, color='r', linestyle='-', linewidth=1, label='Expected Speed')
    plt.tight_layout()
    plt.savefig(f'images/{info_dict["filename"][0]}.png', dpi=300)
    actual_mean_velocity_from_mobicom=np.mean(mobicom_vel_frame_wise)


def plot_range(max_range_index,info_dict):
    plt.figure(figsize=(10, 6))
    plt.plot(max_range_index, label='Range index of brighest range bin', marker='o', markersize=5, linestyle='-', linewidth=1, alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Range index')
    plt.title(f'Range index of brighest range bin {info_dict["filename"][0]}\n pwm value={info_dict[" PWM"][0]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('brightest_range.png')


def get_velocity_antennawise(range_FFT_,peak, info_dict):
        phase_per_antenna=[]
        vel_peak=[]
        for k in range(0,cfg.LOOPS_PER_FRAME):
            r = range_FFT_[k][peak].real
            i = range_FFT_[k][peak].imag
            phase=get_phase(r,i)
            phase_per_antenna.append(phase)
        phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
        cur_vel=solve_equation(phase_cur_frame,info_dict)
        return cur_vel


def get_velocity(rangeResult,range_peaks,info_dict):
    vel_array_frame=[]
    for peak in range_peaks:
        vel_arr_all_ant=[]
        for i in range(0,cfg.NUM_TX):
            for j in range(0,cfg.NUM_RX):
                cur_velocity=get_velocity_antennawise(rangeResult[i][j],peak,info_dict)
                vel_arr_all_ant.append(cur_velocity)
        vel_array_frame.append(vel_arr_all_ant)
    return vel_array_frame

def find_peaks_in_range_Heatmap(rangeHeatmap, pointcloud_processcfg, intensity_threshold):
    range_result_absnormal_split = []
    r_r = rangeHeatmap
    r_r[:,0:10] = 0
    min_val = np.min(r_r)
    max_val = np.max(r_r)
    r_r_normalise = (r_r - min_val) / (max_val - min_val) * 1000
    range_result_absnormal_split.append(r_r_normalise)

    range_abs_combined_nparray = np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame, pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray += ele
    range_abs_combined_nparray /= (pointcloud_processcfg.frameConfig.numTxAntennas * pointcloud_processcfg.frameConfig.numRxAntennas)
    
    range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / pointcloud_processcfg.frameConfig.numLoopsPerFrame
    peaks, _ = find_peaks(range_abs_combined_nparray_collapsed)

    peaks_min_intensity_threshold = []
    for indices in peaks:
        if range_abs_combined_nparray_collapsed[indices] > intensity_threshold:
            peaks_min_intensity_threshold.append(indices)
    
    return peaks_min_intensity_threshold


def find_peaks_in_range_data(rangeResult, pointcloud_processcfg, intensity_threshold):
    range_result_absnormal_split = []
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            r_r = np.abs(rangeResult[i][j])
            r_r[:,0:10] = 0
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * 1000
            range_result_absnormal_split.append(r_r_normalise)

    range_abs_combined_nparray = np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame, pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray += ele
    range_abs_combined_nparray /= (pointcloud_processcfg.frameConfig.numTxAntennas * pointcloud_processcfg.frameConfig.numRxAntennas)
    
    range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / pointcloud_processcfg.frameConfig.numLoopsPerFrame
    peaks, _ = find_peaks(range_abs_combined_nparray_collapsed)

    peaks_min_intensity_threshold = []
    for indices in peaks:
        if range_abs_combined_nparray_collapsed[indices] > intensity_threshold:
            peaks_min_intensity_threshold.append(indices)
    
    return peaks_min_intensity_threshold

def check_consistency_of_frame(current_peaks, next_peaks, threshold):
    if not any(any(abs(c - n) <= threshold for n in next_peaks) for c in current_peaks):
        return False
    return True

def get_consistent_peaks(current_peaks, next_peaks, threshold):
    consistent_peaks = [current_peaks[i] for i, val in enumerate(any(abs(c-n) <= threshold for n in next_peaks) for c in current_peaks) if val]
    return consistent_peaks

def run_data_read_only_sensor(info_dict):
    filename = 'datasets/'+info_dict["filename"][0]
    command =f'python3 data_read_only_sensor.py {filename} {info_dict[" Nf"][0]}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = process.stdout
    stderr = process.stderr

def call_destructor(info_dict):
    file_name="datasets/only_sensor"+info_dict["filename"][0]
    command =f'rm {file_name}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = process.stdout
    stderr = process.stderr


def get_mae(true_vel,doppler_vel,mobicom_vel,info_dict):
    doppler_mae=0
    mobicom_mae=0
    print(f"Doppler vel length = {len(doppler_vel)}")
    print(f"Mobicom vel length = {len(mobicom_vel)}")
    for i in range(len(mobicom_vel)):
        doppler_mae+=np.abs(true_vel/100-doppler_vel[i])
        mobicom_mae+=np.abs(true_vel/100-mobicom_vel[i])
    doppler_mae/=len(doppler_vel)
    mobicom_mae/=len(mobicom_vel)
    df = pd.DataFrame({'pwm': info_dict[' PWM'],'doppler_mae': [doppler_mae], 'mobicom_mae': [mobicom_mae]})
    
    df.to_csv('velocities.csv', mode='a', header=False, index=False)

    true_vel=np.mean(mobicom_vel)
    doppler_mae_array=[]
    mobicom_mae_array=[]
    mode_mobicom_mae_array = []
    for i in range(len(mobicom_vel)):
        doppler_mae_array.append(np.abs(true_vel-doppler_vel[i]))
        mobicom_mae_array.append(np.abs(true_vel-mobicom_vel[i]))
        mode_mobicom_mae_array.append(np.abs(true_vel-mode_velocities[i]))

    fig, ax = plt.subplots()

    box1 = ax.boxplot(doppler_mae_array, positions=[1], widths=0.6, patch_artist=True,medianprops=dict(color="none"),showfliers=False)
    box2 = ax.boxplot(mobicom_mae_array, positions=[2], widths=0.6, patch_artist=True,medianprops=dict(color="none"),showfliers=False)
    box3 = ax.boxplot(mode_mobicom_mae_array, positions=[3], widths=0.6, patch_artist=True,medianprops=dict(color="none"),showfliers=False)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Doppler', 'Mobicom', 'Mode-mobicom'])
    ax.set_title('Box Plot')
    colors = ['lightblue', 'lightgreen', 'pink']
    for box, color in zip([box1, box2, box3], colors):
        for patch in box['boxes']:
            patch.set_facecolor(color)
    plt.grid(True)
    plt.grid(True)
    plt.savefig('box_plot.png')


def plot_phase_heatmap(rangeResult, range_peaks):
    plt.clf()
    phase_heatmap = np.ones((182,256))*10
    for peak in range_peaks:
        phase_per_antenna=[]
        for k in range(0,cfg.LOOPS_PER_FRAME):
            r = rangeResult[0][0][k][peak].real
            i = rangeResult[0][0][k][peak].imag
            phase=get_phase(r,i)
            phase_per_antenna.append(phase)
        phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
        for i in range(0, 182):
            phase_heatmap[i][peak] = phase_per_antenna[i]
    sns.heatmap(phase_heatmap)
    plt.savefig("phase_heatmap.png")


def get_mode_velocity(velocity_array_framewise):
    vel_array_all = []
    for velocity_all_antennas in velocity_array_framewise:
        for velocity in velocity_all_antennas:
            vel_array_all.append(velocity)
    vel_mode = statistics.mode(vel_array_all)
    return vel_mode

# Neural Network Implementation


# Loss calculation including MSE and PINN loss
class SolveEquationLoss(tf.keras.losses.Loss):
    def __init__(self, X_train, L_R_array, mse_weight=0.5):
        super(SolveEquationLoss, self).__init__()
        self.i = 0
        self.X_train = np.squeeze(X_train, axis = -1)#.tolist()
        self.L_R_array = L_R_array
        self.mse_weight = mse_weight

    def call(self, y_true, y_pred):
        # PINN loss calculation
        pointCloudProcessCFG = PointCloudProcessCFG()
        peaks_min_intensity_threshold = find_peaks_in_range_data(self.X_train[self.i], pointCloudProcessCFG, intensity_threshold=100)
        calculated_value = get_velocity(self.X_train[self.i], peaks_min_intensity_threshold, self.L_R_array[self.i])
        self.i = self.i+1
        pinn_loss = tf.reduce_mean(tf.square(calculated_value - y_pred))

        # MSE loss calculation
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Combined PINN loss and MSE loss
        combined_loss = (1 - self.mse_weight) * pinn_loss + self.mse_weight * mse_loss

        return combined_loss


def get_cnn():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(182, 256, 1)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3)
    ], name='cnn2d')
    return model2d

# def get_cnn1d():
#     model1d = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, (8, 2), (2, 1), padding="valid", activation='relu', input_shape=(64, 2, 10)),
#         tf.keras.layers.Conv2D(64, (8, 1), (2, 1), padding="valid", activation='relu'),
#         tf.keras.layers.Conv2D(96, (4, 1), (2, 1), padding="valid", activation='relu'),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(rate=0.3)
#     ], name='cnn1d')
#     return model1d


def get_model():
    cnn = get_cnn()
    # cnn1d = get_cnn1d()
    input = tf.keras.layers.Input(shape=(182, 256, 1))
    # input_2 = tf.keras.layers.Input(shape=(64, 1, 10))
    # input_3 = tf.keras.layers.Input(shape=(64, 1, 10))
    # input_23 = tf.keras.layers.Concatenate(axis=2)([input_2, input_3])
    emb = cnn(input)
    # emb2 = cnn1d(input_23)
    # emb = tf.keras.layers.Concatenate(axis=1)([emb1, emb2])
    output = tf.keras.layers.Dense(units=1, activation='linear')(emb)
    model = tf.keras.Model(inputs=input, outputs=output)
    # model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=output)
    print(model.summary())
    # model.compile(loss=SolveEquationLoss(), optimizer='adam', metrics=['mse'])
    return model


def preprocess_input_cnn(X_train):
    dop_train = np.array(
        [np.concatenate([np.expand_dims(e, 2) for e in v.transpose(1, 0)[2]], axis=2) for v in X_train])
    rp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]], axis=2) for v in X_train])
    noiserp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[1]], axis=2) for v in X_train])

    dop_train_s = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
    rp_train_s = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
    noiserp_train_s = (noiserp_train - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())
    return dop_train_s, rp_train_s, noiserp_train_s

def traincnn(model, X_train, y_train, epochs=500):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        model.compile(loss='mse', optimizer='adam', metrics=["mse"])
        history = \
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_split=0.2,
                batch_size=32,
            )
        plt.plot(history.history['loss'], label='MSE Loss')
        plt.plot(history.history['val_loss'], label='Validation MSE Loss')
        plt.legend()
        plt.show()
        return model


def train(model, X_train, y_train, L_R_array, epochs=500):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    model.compile(loss=SolveEquationLoss(X_train, L_R_array), optimizer='adam', metrics=["mse"])
    X_train = np.abs(X_train)
    X_train  = np.sum(X_train, axis=(0,1))
    X_train = np.expand_dims(X_train, axis = -1)
    history = \
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
        )
    plt.plot(history.history['loss'], label='MSE Loss')
    plt.plot(history.history['val_loss'], label='Validation MSE Loss')
    plt.plot(history.history['SolveEquationLoss'], label='Combined Loss')
    plt.plot(history.history['val_SolveEquationLoss'], label='Validation Combined Loss')
    plt.legend()
    plt.show()
    return model


def test(model, X_test, y_test):
    test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss}')
    print(f'Test MSE: {test_mse}')
    return test_loss, test_mse

def preprocess_input(X_train):
    dop_train = np.array(
        [np.concatenate([np.expand_dims(e, 2) for e in v.transpose(1, 0)[2]], axis=2) for v in X_train])
    rp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[0]], axis=2) for v in X_train])
    noiserp_train = np.array(
        [np.concatenate([np.expand_dims(e.reshape(-1, 1), 2) for e in v.transpose(1, 0)[1]], axis=2) for v in X_train])

    dop_train_s = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
    rp_train_s = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
    noiserp_train_s = (noiserp_train - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())
    return dop_train_s, rp_train_s, noiserp_train_s


# def preprocess_input(X_train):
#     # Initialize empty lists for Doppler, RP, and NoiseRP
#     dop_train = []
#     rp_train = []
#     noiserp_train = []
    
#     # Iterate over each sequence in X_train
#     for sequence in X_train:
#         dop_sequence = np.concatenate([np.expand_dims(frame[:, :, 2], axis=-1) for frame in sequence], axis=-1)
#         rp_sequence = np.concatenate([np.expand_dims(frame[:, :, 0].reshape(-1, 1), axis=-1) for frame in sequence], axis=-1)
#         noiserp_sequence = np.concatenate([np.expand_dims(frame[:, :, 1].reshape(-1, 1), axis=-1) for frame in sequence], axis=-1)
        
#         dop_train.append(dop_sequence)
#         rp_train.append(rp_sequence)
#         noiserp_train.append(noiserp_sequence)
    
#     # Convert lists to numpy arrays
#     dop_train = np.array(dop_train)
#     rp_train = np.array(rp_train)
#     noiserp_train = np.array(noiserp_train)
    
#     # Normalize each array
#     dop_train_s = (dop_train - dop_train.min()) / (dop_train.max() - dop_train.min())
#     rp_train_s = (rp_train - rp_train.min()) / (rp_train.max() - rp_train.min())
#     noiserp_train_s = (noiserp_train - noiserp_train.min()) / (noiserp_train.max() - noiserp_train.min())
    
#     return dop_train_s, rp_train_s, noiserp_train_s


def get_df():
    pkl_file_path = "merged_data.pkl"
    with open(pkl_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict
#     pkl_file_path = "merged_data.pkl"
#     merged_df = pd.read_pickle(pkl_file_path)
#     return merged_df

def get_xtrain_ytrain(merged_df, frame_stack=10):
    # Frame stacking for input features
    X = []
    y = []
    
    for i in range(len(merged_df) - frame_stack + 1):
        range_heatmaps = merged_df.iloc[i:i+frame_stack]['rangeHeatmap'].to_list()
        velocity = merged_df.iloc[i:i+frame_stack-1]['velocity'] 
        
        X.append(range_heatmaps)
        y.append(velocity)
    print(X)
    # Convert lists to numpy arrays
    X = np.asarray(X, dtype=object)
    y = np.array(y)
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test