import numpy as np
import struct
import sys
import os

FRAMES = 50

dca_name = sys.argv[1]
n_frames = int(sys.argv[2])

annotated_fname = "only_sensor"+dca_name
FRAMES = n_frames+1

ADC_PARAMS = {'chirps': 128,  # 32
              'rx': 4,
              'tx': 3,
              'samples': 256,
              'IQ': 2,
              'bytes': 2}

array_size = ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] * ADC_PARAMS['IQ'] * ADC_PARAMS['samples']
element_size = ADC_PARAMS['bytes']


def read_and_print_dca_file(filename, packet_size):
    rows = FRAMES
    cols = (728 * 1536)  # (N_bytes_in_packet x N_packets_in_frame) Integer division

    # Creating a numpy array of uint16 type, initialized with zeros
    frame_array = np.zeros((rows, cols), dtype=np.uint16)
    frame_time_array=np.zeros(FRAMES,dtype=np.float64)
    dirty_array=np.zeros(FRAMES)
    index=0 
    with open(filename,'rb') as file:
        last_packet_num=0
        while True:
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('d',timestamp_data)[0]
            data=file.read(packet_size)
            packet_num=struct.unpack('<1l',data[:4])[0]
            last_packet_num=packet_num
            # byte_count=struct.unpack('>Q',b'\x00\x00'+data[4:10][::-1])[0]
            if (packet_num%(1536))==0:
                print("iske baad se data read chalu karenge")
                print(packet_num)
                break
        
        packet_idx_in_frame=0
        while True:
            
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('d',timestamp_data)[0]
            
            data=file.read(packet_size) # The next packet_data
            if not data:
                break
            packet_num=struct.unpack('<1l',data[:4])[0]
            if packet_num==last_packet_num+1:
                last_packet_num=packet_num
              
                frame_array[index][packet_idx_in_frame:packet_idx_in_frame+728]= np.frombuffer(data[10:], dtype=np.uint16)
                packet_idx_in_frame+=728
                if packet_idx_in_frame==728*1535:
                    frame_time_array[index]=timestamp 
                    packet_idx_in_frame=0
                    index+=1
                continue
            elif packet_num>last_packet_num+1:
                #Packet lost ho gaya hai
                #matlab yeh frame chud gaya hai and we have to reject it
                dirty_array[index]=1
                frame_array[index][packet_idx_in_frame:packet_idx_in_frame+728]=np.zeros(728)
                packet_idx_in_frame+=728
                last_packet_num=packet_num
                if packet_idx_in_frame==728*1535:
                # if packet_num%1536==0:
                    frame_time_array[index]=timestamp
                    packet_idx_in_frame=0
                    index+=1
                continue
                
                # while (packet_num%1536)!=0:
                #     timestamp_data=file.read(8)
                #     data=file.read(packet_size)
                #     if not data:
                #         break
                #     packet_num=struct.unpack('<1l',data[:4])[0]
                # frame_time_array[index]=timestamp_data
                # index+=1
                
            # byte_count=struct.unpack('>Q',b'\x00\x00'+data[4:10][::-1])[0]

            # if c==1:
            #     print(timestamp)
            #     print(packet_num)
            
            # if (byte_count%(728*1536))==0:
            #     print("Hello")
            #     print(packet_num)
    # print(timestamp)
        for i in range(FRAMES):
            if dirty_array[i]==1:
                #reject the frame
                if i==0:
                    j=i
                    while(dirty_array[j]==0):
                        j+=1
                    frame_array[i]=frame_array[j]
                else:
                    j=i
                    while(j>=0 and dirty_array[j]==0):
                        j-=1 
                    frame_array[i]=frame_array[j]
        #Now we have a frame array proper of 100*(1456*1536))
    print("Dirtysum")
    print(np.sum(dirty_array))
    return frame_array,frame_time_array

def annotate(dca_array,frames):

    if os.path.exists(annotated_fname):
        os.remove(annotated_fname)
    annotation_file = open(annotated_fname, "ab")
    for i in range (frames):
        annotation_file.write(dca_array[i])
    annotation_file.close()

dca_array,dca_time_array=read_and_print_dca_file(dca_name,1466)
annotate(dca_array,FRAMES)
print(dca_array.shape)