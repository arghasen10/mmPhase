import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
# Load the JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

max_speed = -56576575
max_speed_i = 0
min_speed = 7867868669
min_speed_i = 0

for i, d in enumerate(data):
    if max_speed <= d["vicon_gt_final"]:
        max_speed = d["vicon_gt_final"]
        max_speed_i = i
        
    if min_speed >= d["vicon_gt_final"]:
        min_speed = d["vicon_gt_final"]
        min_speed_i = i


mse_ours_list = [(x*100 - min_speed)**2 for x in data[min_speed_i]["our"]]
mse_dop_based_list = [(x*100 - min_speed)**2 for x in data[min_speed_i]["dop_based"]]

df2 = pd.DataFrame({"mmPhase": mse_ours_list, "dop_based": mse_dop_based_list, "ground": list(np.ones(len(mse_dop_based_list))*min_speed), "hue": "Low"}) 

print(min_speed, data[min_speed_i]['our'])
mse_ours_list = [(x*100 - max_speed)**2 for x in data[max_speed_i]["our"]]
mse_dop_based_list = [(x*100 - max_speed)**2 for x in data[max_speed_i]["dop_based"]]
# bpl = plt.boxplot(mse_ours_list, positions=np.array(range(len(mse_ours_list)))*2.0-0.4, sym='', widths=0.6)

# bpr = plt.boxplot(mse_dop_based_list, positions=np.array(range(len(mse_dop_based_list)))*2.0+0.4, sym='', widths=0.6)
df1 = pd.DataFrame({"mmPhase": mse_ours_list, "dop_based": mse_dop_based_list, "ground": list(np.ones(len(mse_dop_based_list))*max_speed), "hue": "High"})
df = pd.concat([df1, df2], axis=0)

sns.boxplot(x='hue', y='dop_based', hue='ground', data=df)
plt.ylabel("MSE", size=12)
plt.legend(loc='upper right')
plt.savefig('test.png')
