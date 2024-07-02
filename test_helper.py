import tensorflow as tf
import pandas as pd
import numpy as np
import os
from helper import PointCloudProcessCFG, get_velocity, find_peaks_in_range_data

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class CustomModel(tf.keras.Model):
    def __init__(self, mse_weight=0.5):
        super(CustomModel, self).__init__()
        self.mse_weight = mse_weight
        self.cnn2d = self.build_cnn2d()

    def build_cnn2d(self):
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

    def call(self, inputs):
        rangeResult, velocity, L_R = inputs
        rangeResult = np.sum(rangeResult, axis=1)
        y_true = velocity
        y_estimate = []
        for i in range(len(L_R)):
            L_R_dict = {' L':L_R[i][0], ' R':L_R[i][1]}
            pointCloudProcessCFG = PointCloudProcessCFG()
            range_peaks = find_peaks_in_range_data(rangeResult[i], pointCloudProcessCFG, intensity_threshold=100)
            estimated_velocity = get_velocity(rangeResult[i], range_peaks, L_R_dict)
            estimated_velocity = np.array(estimated_velocity)
            y_estimate.append(estimated_velocity.mean())
        y_estimate = tf.convert_to_tensor(y_estimate, dtype=tf.float32)
        
        #
        rangeResult = np.expand_dims(rangeResult,-1)
        rangeResultabs = np.abs(rangeResult)
        rangeResultabs = np.sum(rangeResultabs, axis=(1,2))
        rangeResultsum = tf.convert_to_tensor(rangeResultabs, dtype=tf.float32) 
        y_pred = self.cnn2d(rangeResultsum)
        return y_pred, y_estimate, y_true
        

    def compute_loss(self, y_pred, y_true, y_estimate):
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        pinn_loss = tf.reduce_mean(tf.square(y_estimate - y_pred)) # changed - y_true to - y_pred
        combined_loss = (1 - self.mse_weight) * pinn_loss + self.mse_weight * mse_loss
        return tf.math.reduce_mean(combined_loss)

mse_weight = 0.5  
model = CustomModel(mse_weight=mse_weight)

optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

def train_step(model, inputs):
    with tf.GradientTape() as tape:
        y_pred, y_estimate, y_true = model(inputs)
        loss = model.compute_loss(y_pred, y_true, y_estimate)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


print("############Start Data Loading############")
df=pd.read_pickle('merged_data.pkl')
print("############Data Loaded############")
rangeResult=np.expand_dims(df['rangeResult'],1)
L_R=df['L_R']
velocity=df['velocity']

def get_batch(size=64):
    rand_idx=np.random.randint(0,L_R.shape[0],size)
    return rangeResult[rand_idx],velocity[rand_idx],L_R[rand_idx]

epochs = 100
steps_per_epoch = 512

for epoch in range(epochs):
    epoch_loss=0
    for s_e in range(steps_per_epoch):
        batch=get_batch()
        loss = train_step(model, batch)
        if (s_e+1)%10==0:
            print(f"s_e {s_e+1}, Loss: {loss.numpy()}")
        epoch_loss+=loss.numpy()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

    # Save checkpoint every epoch
    save_path = manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch + 1, save_path))
