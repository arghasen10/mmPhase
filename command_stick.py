import serial
import time
import subprocess
import sys
import os
import csv
from datetime import datetime
import argparse
import cv2
# import board 
# import adafruit_mpu6050
import threading
from utils.imu_data_collector import collect_data
from utils.video_cap import capture_video
#from git import Repo
#from utils import push

def execute_c_program(c_program_path, c_program_args):
    command=[c_program_path] + c_program_args
    print("command: ", command)
    # Execute the C program
    try:
        print("Executing C program...")
        result = subprocess.run(command, check=True)
        print("C program executed successfully.")
    except subprocess.CalledProcessError as e:
        #print(f"Error executing C program: {e}")
        pass

def capture_frame_and_save(folder_path, image_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera, exiting thread")
        sys.exit()
        return
    ret, frame = cap.read()
    cap.release()
    if ret:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, frame)
        print("Image saved successfully:", image_path)
    else:
        print("Error: Failed to capture frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for params')
    parser.add_argument('-nf', '--nframes', type=int, help='Number of frames')
    parser.add_argument('-nc', '--nchirps', type=int, help='Number of chirps in a frame, usually 182')
    parser.add_argument('-tc', '--timechirp', type=int, help='Chrip time is microseconds, usually 72')
    parser.add_argument('-s', '--samples', type=int, help='Number of ADC samples, or range bins, usually 256')
    parser.add_argument('-r', '--rate', type=int, help='Sampling rate, usually 4400')
    parser.add_argument('-tf', '--timeframe', type=int, help='Periodicity or Frame time in milliseconds')
    parser.add_argument('-l', '--length', type=int, help='Initial length')
    parser.add_argument('-r0', '--radial', type=int, help='Initial radial distance')
    parser.add_argument('-d', '--descp', type=str, help='Data description')
    parser.add_argument('-camera', action='store_true')
    parser.add_argument('-imu', action='store_true')
    ans1=input("Have you connected the ethernet to Jetson? yes/no: ")
    camera_pass = False
    args = parser.parse_args()
    mac_command = f"sudo macchanger --mac=08:97:98:70:B9:13 eth0"
    print(mac_command)
    os.system(mac_command)
    if(args.camera):
        ans3=input("Have you connected camera cable? yes/no: ")
        if(ans3=="yes"):
            camera_pass = True
    elif(not args.camera):
        camera_pass= True
    if ans1=='yes' and camera_pass: 
        c_program_path = "/home/stick/mmPhase/data_collect_mmwave_only" 
        image_folder_path = "./scene_annotation/"
        now = datetime.now()
        date_string = str(now.strftime('%Y-%m-%d_%H_%M_%S'))
        n_frames = str(args.nframes)
        n_chirps = str(args.nchirps)
        tc       = str(args.timechirp)
        adc_samples = str(args.samples)
        sampling_rate = str(args.rate)
        periodicity = str(args.timeframe)
        l = str(args.length)
        r0 = str(args.radial)
        descri = args.descp
        date_string+="_" + descri
        file_name="stick_"+date_string+"_"+".bin"
        image_name = "stick_"+date_string+"_"+".jpg"
        c_program_args=[file_name,n_frames]
        if(args.camera):
            capture_frame_and_save(image_folder_path, image_name)
        # video_filename =  date_string+"_"+pwm_value+".mp4"
        # video_thread = threading.Thread(target=capture_video, args=(imu_duration, video_filename))
        # video_thread.start()
        if(args.imu):
            imu_duration = (int(n_frames)+5)*int(periodicity) / 1000; #periodicity is in ms (collect for 5 extra frames)
            imu_filename = "stick_"+date_string+"_imu.bin"
            imu_thread = threading.Thread(target=collect_data, args=(imu_duration, imu_filename))
            imu_thread.start()     
        execute_c_program(c_program_path,c_program_args)
        if(args.imu):
            imu_thread.join()    
            # video_thread.join()
        ans_to_keep=input('Do you want to keep the reading? yes/no : ')
        if(ans_to_keep=='no'):
            os.system(f"rm {file_name}")
            print(f"{file_name} deleted successfully")
            os.system(f"rm ./imu_data/{imu_filename}")
            print(f"./imu_data/{imu_filename} deleted successfully")
            sys.exit()
        #os.system(f"mv {file_name} /media/stick/Seagate\ Backup\ Plus\ Drive/")
        #if (args.imu):
            #os.system(f"mv ./imu_data/{imu_filename} /media/stick/Seagate\ Backup\ Plus\ Drive/imu_data/")
        file_path="dataset_stick.csv"
        data=[file_name,n_frames,n_chirps,tc,adc_samples,sampling_rate,periodicity,l,r0,descri]
        if r0==l:
            data.append('Straight')
        else:
            data.append('Oblique')
        
        with open(file_path,'a',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(data)
            print('Data appended successfully')

