import pandas as pd
import os
import subprocess

dataset=pd.read_csv('dataset.csv')
if not os.path.exists('mat.csv'):
    with open('mat.csv', 'w') as f:
        f.write('Mat file name,speed\n')
# print(dataset.columns)
for index, row in dataset.iterrows():
    if index==0:
        continue
    # Get the filename from the 'file_name' column
    file_name = row['filename']
    num_frames=row[' Nf']
    print(file_name,num_frames)
    if os.path.exists(file_name):
        os.system(f'python3 data_read_only_sensor.py {file_name} {num_frames}')
        command = "python3"
        script = "phase_generation.py"
        file_name='only_sensor'+file_name
        os.system(f'python3 phase_generation.py {file_name} {num_frames}')
# Run the command and capture the output
#         result = subprocess.run(args, shell=True, check=True, stdout=subprocess.PIPE, text=True)

# # Extract the output (number)
        
#         idx=result.stdout.split('\n')[0]

        with open('mat.csv', 'a') as mat_file:
            only_sensor_filename = f'only_sensor{file_name}'[:-4]+'.mat'
            speed = row[' Vb']  # Assuming 'speed' is a column in your dataset
            mat_file.write(f'{only_sensor_filename},{speed}\n')

