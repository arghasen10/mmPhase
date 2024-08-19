import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from trajectory_modified import Trajectory
import math
random.seed(42)
np.random.seed(42)
#store the objects with their coordinates in a global_coordinates dict
#Let's now simulate the value
global_coordinates={'ID1':(20,25),'ID2':(-12,17)}
#create an artificial path of 5 steps
start=(0,0)
x_movs=[np.random.randint(-7,7) for i in range(10)]
y_movs=[np.random.randint(0,7) for i in range(10)]
# Calculate the trajectory from the start point
trajectory_x = [start[0]] + np.cumsum(x_movs).tolist()
trajectory_y = [start[1]] + np.cumsum(y_movs).tolist()
#initialise the object
track=Trajectory()

percentage_error_in_distance=0.02
percentage_error_in_velocity=0.05
for i in range(len(trajectory_x)):
    if i==0:
        continue
    global_xcor=trajectory_x[i]
    global_ycor=trajectory_y[i] #in reality we wont be having this
    local_coordinates={}
    for obj in global_coordinates.keys():
        rel_xcor1=(global_xcor-global_coordinates[obj][0])*(1+np.random.choice([-1, 1])*percentage_error_in_distance)
        rel_ycor1=(global_ycor-global_coordinates[obj][1])*(1+np.random.choice([-1, 1])*percentage_error_in_distance)
        local_coordinates[obj]=(rel_xcor1,rel_ycor1)#This is what we are supposed to extract from the mmwave radar
    #create an array of statis objects
    '''
         
        ---      ---
        |x1, y1, r1|
        |x2, y2, r2| 
        |    .     |
        |    .     |
        |xn, yn, rn|
        ---      ---
    '''
    #calculated the simulated velocity
    v_sim=math.sqrt((trajectory_x[i]-trajectory_x[i-1])**2+(trajectory_y[i]-trajectory_y[i-1])**2)*(1+np.random.choice([1,-1])*percentage_error_in_velocity)
    static_objects=[[global_coordinates[obj][0],global_coordinates[obj][1],float(np.sqrt(np.square(local_coordinates[obj][0])+np.square(local_coordinates[obj][1])))] for obj in local_coordinates.keys()]
    track.estimate(static_objects,v_sim)
predicted_trajectory_x=track.traj[:,0]
predicted_trajectory_y=track.traj[:,1]
print("Actual trajectory")
print(trajectory_x,trajectory_y)
print("Predicted trajectory")
print(predicted_trajectory_x,predicted_trajectory_y)
plt.plot(trajectory_x, trajectory_y, marker='o', linestyle='-', color='blue', label='Actual Trajectory')

# Plotting the predicted trajectory
plt.plot(predicted_trajectory_x, predicted_trajectory_y, marker='x', linestyle='--', color='red', label='Predicted Trajectory')
plt.scatter(trajectory_x[0], trajectory_y[0], color='green', marker='o', s=100, label='Start Point (Actual)')
plt.scatter(predicted_trajectory_x[0], predicted_trajectory_y[0], color='purple', marker='x', s=100)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(
    f'Comparison of Actual and Predicted Trajectories: '
    f'{percentage_error_in_distance * 100:.2f}% error in range estimation, '
    f'{percentage_error_in_velocity * 100:.2f}% error in velocity estimation'
)
plt.grid(True)
plt.legend()
plt.savefig('trajectory_comparison.png')
plt.show()

