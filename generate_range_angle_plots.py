from import_all import *
import glob
import pickle

def read8byte(x):
    return struct.unpack('<hhhh', x)


class FrameConfig:  #
    def __init__(self):
        #  configs in configuration.py
        self.numTxAntennas = cfg.NUM_TX
        self.numRxAntennas = cfg.NUM_RX
        self.numLoopsPerFrame = cfg.LOOPS_PER_FRAME
        self.numADCSamples = cfg.ADC_SAMPLES
        self.numAngleBins = cfg.NUM_ANGLE_BINS

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame

        # calculate size of one chirp in short.
        self.chirpSize = self.numRxAntennas * self.numADCSamples
        # calculate size of one chirp loop in short. 3Tx has three chirps in one loop for TDM.
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        # calculate size of one frame in short.
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

        # 0,1,2 for x,y,z
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


def dopplerFFT(rangeResult, frameConfig):  #
    windowedBins2D = rangeResult * np.reshape(np.hamming(frameConfig.numLoopsPerFrame), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult


def get_coordinates(dopplerResult):
    #First 30cm make it very negative so the first 3 bins
    cfar_result=np.zeros(dopplerResult.shape,bool)
    top_128=128
    energy_threshold = np.partition(dopplerResult.ravel(), 182 * 256 - top_128 - 1)[182 * 256 - top_128 - 1]
        #So energy Thre128 is the 128th most energetic point
    # print(energy_threshold)
    cfar_result[dopplerResult>energy_threshold]=True
    det_peaks_indices = np.argwhere(cfar_result == True)
    # print(det_peaks_indices.shape)
    object_energy_coordinates=np.zeros((top_128,3))
    object_energy_coordinates[:,0]=det_peaks_indices[:,0]
    object_energy_coordinates[:,1]=det_peaks_indices[:,1]
    for i in range(top_128):
        x_cor=object_energy_coordinates[i][0]
        y_cor=object_energy_coordinates[i][1]
        object_energy_coordinates[i][2]=dopplerResult[int(x_cor)][int(y_cor)]
    
    return object_energy_coordinates,cfar_result
        

def get_azimuthal_angle(dopplerResult,cfar_result):
    az_angle_map={}
    for i in range(cfar_result.shape[0]):
        for j in range(cfar_result.shape[1]):
            if cfar_result[i][j]==True:
                key=(i,j)
                az_angle_map[key]=dopplerResult[:,:,i,j].reshape(12,-1).flatten()[0:8]
    for key,value in az_angle_map.items():
        azimuth_fft_padded=np.zeros(64,dtype=np.complex_)
        azimuth_fft_padded[0:8]=az_angle_map[key]
        azimuth_fft_padded=np.fft.fft(azimuth_fft_padded)
        azimuth_fft_padded = np.fft.fftshift(azimuth_fft_padded)
        az_angle_map[key]=np.abs(azimuth_fft_padded)
    
    return az_angle_map


def get_args():
    parser=argparse.ArgumentParser(description="Run the phase_generation script")
    parser.add_argument('-f','--file_name',help="Get the .bin file to process")
    args=parser.parse_args()
    return args


def get_info(f):
    dataset=pd.read_csv('dataset.csv')
    file_name=f
    filtered_row=dataset[dataset['filename']==file_name]
    info_dict={}
    for col in dataset.columns:
        info_dict[col]=filtered_row[col].values
    if len(info_dict['filename'])==0:
        print('Oops! File not found in database. Cross check the file name')
    else:
        print('Great! Your file has been found in our dataset')
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


def run_data_read_only_sensor(info_dict):
    filename = 'datasets/'+info_dict["filename"][0]
    command =f'python3 data_read_only_sensor.py {filename} {info_dict[" Nf"][0]}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout)
    print('Data_read_only_sensor.py executed successfully')

def call_destructor(info_dict):
    file_name = 'datasets/only_sensor'+info_dict["filename"][0]
    command =f'rm {file_name}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = process.stdout
    stderr = process.stderr

    
    
def collect_ra_heatmap(f):
    info_dict=get_info(f.split("/")[-1])
    print_info(info_dict)
    stdout = run_data_read_only_sensor(info_dict)
    # print(stdout)
    bin_filename = 'datasets/'+info_dict["filename"][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = info_dict[' Nf'][0]
    pointCloudProcessCFG = PointCloudProcessCFG()
    collect_range_angle = []
    for frame_no in tqdm(range(total_frame_number)):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        dopplerResult = dopplerFFT(rangeResult, frameConfig)      
        dopplerResultabs=np.absolute(dopplerResult)
        dopplerResultabs=np.sum(dopplerResultabs,axis=(0,1))
        _,cfar_result=get_coordinates(dopplerResultabs)
        az_angle_map=get_azimuthal_angle(dopplerResult,cfar_result)
        range_angle=np.zeros((256,64),dtype=np.complex_)
        for key,value in az_angle_map.items():      
            range_angle[key[1]]+=np.abs(value)           
        collect_range_angle.append(range_angle)

    bin_reader.close()  
    call_destructor(info_dict)
    print(info_dict)
    return np.array(collect_range_angle)

def read_imu(f):
    full_path = "imu_data/"+f.split("/")[-1].split(".")[0]+"_imu.bin"
    imu_datas = []
    timestamps = []
    with open(full_path, 'rb') as file:
        # Read and unpack the binary data
        while True:
            # Read 8 bytes for the timestamp and 48 bytes for the IMU data (6 values * 8 bytes each)
            packed_data = file.read(8)
            if not packed_data:
                break  # End of file reached
            
            # Unpack the data into a timestamp and IMU data
            timestamp = struct.unpack('d' , packed_data)
            timestamps.append(timestamp)
            packed_data = file.read(48)
            imu_data = struct.unpack('d' * 6 , packed_data)
            imu_datas.append(imu_data)
    return np.array(timestamps), np.array(imu_datas)


def read_sensor_timestamp(f):
    full_path = "time_stamps/time"+f.split("/")[-1].split(".")[0]+".bin"
    timestamps = []
    with open(full_path, 'rb') as file:
        # Read and unpack the binary data
        while True:
            # Read 8 bytes for the timestamp and 48 bytes for the IMU data (6 values * 8 bytes each)
            packed_data = file.read(8)
            if not packed_data:
                break  # End of file reached
            
            # Unpack the data into a timestamp and IMU data
            timestamp = struct.unpack('d' , packed_data)
            timestamps.append(timestamp)
    print("Sensor timestamps: ", timestamps)
    return np.array(timestamps)

if __name__ == '__main__':
    for f in glob.glob("datasets/*.bin"):
        collect_range_angle = collect_ra_heatmap(f)
        imu_timestamps, imudata = read_imu(f)
        file_name = "milliEgo/"+f.split(".")[0]+".pickle"
        sensor_timestamps = read_sensor_timestamp(f)
        arrays = [collect_range_angle, imudata, sensor_timestamps, imu_timestamps]
        with open(file_name, 'wb') as f:
            pickle.dump(arrays, f)  