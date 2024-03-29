import glob
import os
import subprocess
import tqdm
directory_path = '/home/soham/Desktop/BTP/'
pattern = os.path.join(directory_path, '*.bin')
bin_files = glob.glob(pattern)
for file_path in tqdm.tqdm(bin_files):
    
    file_name=file_path.split('/')[-1]
    print(file_name)
    command = f'python phase_generation_new.py -f {file_name}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

