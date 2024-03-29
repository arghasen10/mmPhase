import sys
import os
import struct
import time
import numpy as np
import array as arr
import configuration as cfg
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import cv2
import subprocess
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
max_range_index=[]
all_range_index=[]
max_doppler_index=[]
all_doppler_index=[]
velocity_array=[]