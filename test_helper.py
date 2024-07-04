import tensorflow as tf
import pandas as pd
import numpy as np
import os
import concurrent.futures
# from tqdm import tqdm
from helper import PointCloudProcessCFG, get_velocity, find_peaks_in_range_data
np.random.seed(42)
tf.random.set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#Training
df=pd.read_pickle('merged_data_quant.pkl')
rangeResultabs=df['rangeResult'].astype(np.float32)
y_true=df['velocity'].astype(np.float32)
y_est=df['y_estimate'].astype(np.float32)

y_TRUE=np.hstack((y_est.reshape(-1,1),y_true.reshape(-1,1)))

dataset=tf.data.Dataset.from_tensor_slices((rangeResultabs,y_TRUE)).shuffle(512).repeat(-1).batch(64,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

def build_cnn2d():
    model2d = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (2, 5), (1, 2), padding="same", activation='relu', input_shape=(182, 256, 1)),
        tf.keras.layers.Conv2D(64, (2, 3), (1, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same", activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(units=1, activation='linear')
    ], name='cnn2d')
    return model2d

@tf.function
def loss_fn(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return mse_loss

model = build_cnn2d()
# optimizer = tf.keras.optimizers.Adam()
model.compile(loss=loss_fn,optimizer='adam')

history=model.fit(dataset,epochs=100,steps_per_epoch=512)
