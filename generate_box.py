import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from seaborn import color_palette

colors = color_palette(n_colors=10)
plt.rcParams.update({'font.size': 28})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# def extract_filenames_and_data(data, data_key, op):
#     filenames_data = {}
#     if op == 0:
#         for entry in data:
#             filenames_data[entry['filename']] = entry[data_key]
#     elif op == 1:
#         for entry in data:
#             filename = '/'.join(entry['filename'].split('/')[1:]).split('.')[0]+'.bin'
#             filenames_data[filename] = entry[data_key]
#     else:
#         for entry in data:
#             filename = entry['filename'].split('.')[0]+'.bin'
#             filenames_data[filename] = entry[data_key]
#     return filenames_data


# with open('estimated_speed_doppler.json', 'r') as dop_file, open('estimated_speed_mmPhase.json', 'r') as mmPhase_file, open('gt_speed.json', 'r') as gt_file, open('imuestimate.json', 'r') as imu_file, open('milliEgo/milliegoestimate.json', 'r') as milliego_file:
#     gt = json.load(gt_file)
#     mmPhase = json.load(mmPhase_file)
#     dop = json.load(dop_file)
#     imu = json.load(imu_file)
#     milliego = json.load(milliego_file)

# gt_vals = extract_filenames_and_data(gt, 'gt_speed', 0)
# mmphase_vals = extract_filenames_and_data(mmPhase, 'estimated_speed', 0)
# dop_vals = extract_filenames_and_data(dop, 'estimated_speed', 0)
# imu_vals = extract_filenames_and_data(imu, 'imu_estimate', 0)
# imu_vals = extract_filenames_and_data(imu, 'imu_estimate', 1)
# milliego_vals = extract_filenames_and_data(milliego, 'estimate', 2)
# # print(milliego_vals)

# common_filenames = set(gt_vals.keys()) & set(mmphase_vals.keys()) & set(dop_vals.keys()) & imu_vals.keys() & set(milliego_vals.keys())

# captured_data = {}
# for filename in common_filenames:
#     captured_data[filename] = {
#         'gt': gt_vals.get(filename),
#         'mmphase': mmphase_vals.get(filename),
#         'dop': dop_vals.get(filename),
#         'imu': imu_vals.get(filename),
#         'milliego': milliego_vals.get(filename),
#     }

# with open('final_result.json', 'w') as file:
#     json.dump(captured_data,file)


with open('final_result.json', 'r') as file:
    captured_data = json.load(file)


pwms = []
low_vel_files = []
mid_vel_files = []
high_vel_files = []
pwms = []
for f in list(captured_data.keys()):
    pwms.append(int(f.split('.')[-2].split('_')[-1]))
pwms = sorted(pwms)
split_idx1 = len(pwms) // 3
split_idx2 = split_idx1 * 2

# Split the array into 3 groups
group1 = pwms[:(split_idx1-4)]
group2 = pwms[(split_idx1+4):(split_idx2-4)]
group3 = pwms[(split_idx2+4):]
for f in list(captured_data.keys()):
    pwm_val = int(f.split('.')[-2].split('_')[-1])
    if pwm_val in group1:
        low_vel_files.append(f)
    elif pwm_val in group2:
        mid_vel_files.append(f)
    else:
        high_vel_files.append(f)
    


low_res = {}
for l in low_vel_files:
    for key in captured_data[l].keys():
        abs_diff = np.abs(np.array(np.abs(captured_data[l][key]))-captured_data[l]['gt'])
        if key == 'gt':
            continue
        if key == 'imu':
            abs_diff = abs_diff *0.02
        if key == 'milliego':
            abs_diff = abs_diff*0.01
        if key not in low_res:
            low_res[key] = []
        low_res[key].append(np.mean(abs_diff))

mid_res = {}
for l in mid_vel_files:
    for key in captured_data[l].keys():
        abs_diff = np.abs(np.array(np.abs(captured_data[l][key]))-captured_data[l]['gt'])
        if key == 'gt':
            continue
        if key == 'imu':
            abs_diff = abs_diff *0.02
        if key == 'milliego':
            abs_diff = abs_diff*0.01
        if key not in mid_res:
            mid_res[key] = []
        mid_res[key].append(np.mean(abs_diff))


high_res = {}
for l in high_vel_files:
    for key in captured_data[l].keys():
        abs_diff = np.abs(np.array(np.abs(captured_data[l][key]))-captured_data[l]['gt'])
        if key == 'gt':
            continue
        if key == 'imu':
            abs_diff = abs_diff*0.02
        if key == 'milliego':
            abs_diff = abs_diff*0.01
        if key not in high_res:
            high_res[key] = []
        high_res[key].append(np.mean(abs_diff))

low_data = [low_res['mmphase'], low_res['dop'], low_res['imu'], low_res['milliego']]
mid_data = [mid_res['mmphase'], mid_res['dop'], mid_res['imu'], mid_res['milliego']]
high_data = [high_res['mmphase'], high_res['dop'], high_res['imu'], high_res['milliego']]

ticks = ['mmPhase', 'Doppler', 'IMU', 'Pretrained\nmilliEgo']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color='k')
    plt.setp(bp['caps'], color='k')
    plt.setp(bp['medians'], color='tab:red')

# plt.figure()

print(np.array(range(len(low_data)))+4*1)
print(np.array(range(len(mid_data)))+4*3)
print(np.array(range(len(mid_data)))+4*5)


p1 = plt.boxplot(low_data, positions=[2,7,12,17], widths=0.6, showfliers=False, patch_artist=True, labels=['mmPhase', 'Doppler', 'IMU', 'Pretrained\nmilliEgo'])
set_box_color(p1, 'tab:blue')
p2 = plt.boxplot(mid_data, positions=[3,8,13,18], widths=0.6, showfliers=False, patch_artist=True, labels=['mmPhase', 'Doppler', 'IMU', 'Pretrained\nmilliEgo'])
set_box_color(p2, 'tab:orange')
p3 = plt.boxplot(high_data, positions=[4,9,14,19], widths=0.6, showfliers=False, patch_artist=True, labels=['mmPhase', 'Doppler', 'IMU', 'Pretrained\nmilliEgo'])
set_box_color(p3, 'tab:green')

plt.legend(handles=[Patch(facecolor=colors[i],label=g,edgecolor='k',linewidth=0.5) for i,g in enumerate(['Low', 'Mid', 'High'])],
          ncols=3,frameon=False,loc='upper left',fontsize=26)
plt.xticks([3, 8, 13, 18], ticks,rotation=15)
plt.grid(axis='y')
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('tab:blue')
leg.legendHandles[1].set_color('tab:orange')
leg.legendHandles[2].set_color('tab:green')
plt.yticks([0,0.2, 0.4, 0.6])
plt.ylabel('MAE (m/s)')
plt.tight_layout()
plt.ylim([0,0.8])
plt.savefig('box_mae.pdf')
plt.show()