import pandas as pd
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
from tqdm import tqdm
import seaborn as sns
import random
import scipy
import os
from collections import defaultdict

class Point:
    threshold=None

    def set_threshold(th):
        Point.threshold=th
    
    def __init__(self, x, y, z, f_id):
        self.x = x
        self.y = y
        self.z = z
        self.frame_id = f_id
    def __rshift__(self,other):
        return np.sqrt((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)

    def __eq__(self,other):
        assert Point.threshold!=None,"set threshold first"
        if self.__rshift__(other)<Point.threshold:
            # print(self.__rshift__(other))
            return True
        else:
            return False
        
    def __repr__(self):
        return f'<{self.x},{self.y},{self.z},{self.frame_id}>'


class Velocity:
    def __init__(self,v,ts):
        self.v=v
        self.ts=ts
    def __repr__(self):
        return f'({self.v},{self.ts})'
    

def preproces_vicon(filename = 'ground_truth/29_03_24_vicon_85_Trajectories_100.csv'): 
    counter = 0
    smooth_file = 'ground_truth/smooth_'+filename.split('/')[-1].split('.')[0]+'.csv'
    with open(smooth_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        with open(filename, 'r') as file:
            lines = csv.reader(file)
            for l in lines:
                counter+=1
                if counter == 3:
                    v = ''
                    l1 = []
                    for elem in l:
                        if elem != '':
                            v=elem
                        l1.append(v)
                if counter == 4:
                    header = []
                    for i, c in enumerate(l):
                        header.append(l1[i]+c)
                    csvwriter.writerow(header)
                if counter >=6:
                    if ''.join(l[1:]) == '':
                        continue
                    else:
                        csvwriter.writerow(l)
    df = pd.read_csv(smooth_file, index_col='Frame')
    os.remove(smooth_file)
    return df, header

def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_gt_velocity(filename):
    df, header = preproces_vicon(filename)
    df.drop('Sub Frame', axis=1, inplace=True)

    

    vicon_data = defaultdict(lambda:[])

    for i in range(df.shape[0]):
        for col in df.columns:
            if 'X' in col:
                y_col = header[2:].index(col)
                if np.isnan(df.iloc[i,y_col]):
                    continue
                vicon_data[i].append(Point(df.iloc[i,y_col],df.iloc[i,y_col+1], df.iloc[i,y_col+2], i))

    prev = None
    ids = {}
    counter = 0
    for e in range(0,len(vicon_data)):
        if prev == None:
            prev = vicon_data[e]
            continue
        for i in vicon_data[e]:
            for p in prev:
                if p.__rshift__(i) < 10:
                    found = False
                    if len(ids) == 0:
                        ids[counter] = []
                        ids[counter].append(i)
                        counter += 1
                        continue
                    for key, val in ids.items():
                        if val[-1].__rshift__(i) < 10:
                            ids[key].append(i)
                            found = True
                    if found == False:
                        ids[counter] = []
                        ids[counter].append(i)
                        counter+=1
        prev = vicon_data[e]


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    color=sns.color_palette(n_colors=len(ids))

    for id in ids.keys():
        if len(ids[id])< 10:
            continue
        x=[p.x for p in ids[id]]
        y=[p.y for p in ids[id]]
        z=[p.z for p in ids[id]]
        ax.scatter(x, y, z,color=color[id],label=f'Marker {id}')

    x=[p.x for fn in range(len(ids)) for p in ids[fn]]
    y=[p.y for fn in range(len(ids)) for p in ids[fn]]
    z=[p.z for fn in range(len(ids)) for p in ids[fn]]
    ax.scatter(x, y, z,color='k',label='Whole',s=1,alpha=0.1)

    plt.legend()
    plt.show()




    dist_unit= 1/10 #cm 1mm
    sampling_freq=100 #hz

    speed_dict={id:[Velocity((((s>>e)*dist_unit))/((e.frame_id-s.frame_id)*(1/sampling_freq)),e.frame_id) for s,e in zip(ids[id][:-1],ids[id][1:])] for id in ids}

    dfff=\
    pd.concat([pd.DataFrame([[le.v,le.ts] for le in speed_dict[id]],columns=[f"vel_{id}","ts"]).set_index('ts') for id in speed_dict],
            join='outer').reset_index().fillna(0).groupby('ts').sum()

    series=dfff.sum(axis=1)/(dfff>0).sum(axis=1)
    df_vel=pd.DataFrame(np.vstack([np.array(list(series.index))/sampling_freq,series.values]).T,columns=['timestamp','vel'])



    sns.kdeplot(df_vel['vel'])
    plt.axvline(np.median(df_vel['vel']),color='k',ls='--',label=f"Median: {np.round(np.median(df_vel['vel']),4)} cm/sec")
    plt.axvline(np.mean(df_vel['vel']),color='green',label=f"Mean: {np.round(np.mean(df_vel['vel']),4)} cm/sec")
    plt.legend()
    plt.savefig('mean_median_speed_estimate.pdf')
    plt.show()

    df_vel['sec']=np.round(df_vel.timestamp)

    plt.figure(figsize=(15,3))
    df_vel.groupby('sec')['vel'].mean().plot(label='mean')
    df_vel.groupby('sec')['vel'].median().plot(label='median')
    plt.ylabel('Velocity (cm/sec)')
    plt.legend()
    plt.ylabel('Vicon Estiated Speed')
    plt.xlabel('No. of Frames')
    plt.savefig('vel_vicon.pdf')
    plt.show()
    return df_vel.groupby('sec')['vel'].mean()

filename = 'ground_truth/29_03_24_vicon_85_Trajectories_100.csv'
print(get_gt_velocity(filename))