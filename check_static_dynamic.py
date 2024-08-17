import numpy as np
from sklearn.cluster import DBSCAN

def find_static(radar_cube_4D):
    '''
    @summary: finds the static points from the 4D radar cube
    @input: radar_cube_4D : [rj, theta_j, phi_j, vj]
    @output: radar cube of static points
    '''
    val_array = np.empty((0,1), np.float32)

    for (r, theta, phi, v) in radar_cube_4D[0]:
        val_array = np.vstack((val_array, v/(np.cos(theta)*np.cos(phi))))
    
    #DBSCAN clustering
    val_array = val_array.reshape(-1, 1)

    dbscan = DBSCAN(eps=.5, min_samples=3)
    labels = dbscan.fit_predict(val_array)
    static_points = radar_cube_4D[val_array[labels == 0]]

    return static_points
