import tensorflow as tf
import numpy as np
import os
import pandas as pd
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
        A, B, C = inputs
        y_true = C

        y_estimate = C #self.get_velocity(A, B)
        
        #
        A_abs = tf.math.abs(A)
        A_abs = tf.transpose(A_abs,perm=(0,2,3,4,5,1))
        A_sum = tf.experimental.numpy.sum(A_abs, axis=1)
        A_sum = tf.experimental.numpy.sum(A_sum, axis=1)
        
        y_pred = self.cnn2d(A_sum)
        
        return y_pred, y_estimate, y_true

    def get_velocity(self, A, B):
        
        return B[:,1]
    def compute_loss(self, y_pred, y_true, y_estimate):
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        pinn_loss = tf.reduce_mean(tf.square(y_estimate - y_true))
        combined_loss = (1 - self.mse_weight) * pinn_loss + self.mse_weight * mse_loss
        return tf.math.reduce_mean(combined_loss)

mse_weight = 0.5  
model = CustomModel(mse_weight=mse_weight)

optimizer = tf.keras.optimizers.Adam()

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
    return rangeResult[rand_idx],L_R[rand_idx],velocity[rand_idx]

# dataset=tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(np.expand_dims(df['rangeResult'],1)),tf.convert_to_tensor(df['L_R']),tf.convert_to_tensor(df['velocity'])]).shuffle().batch(32).repeat()
# print("############DataSet Created############")
epochs = 10
steps_per_epoch=512

for epoch in range(epochs):
    epoch_loss=0
    for s_e in range(steps_per_epoch):
        batch=get_batch()
        loss = train_step(model, batch)
        if s_e%10==0:
            print(f"s_e {s_e+1}, Loss: {loss.numpy()}")
        epoch_loss+=loss.numpy()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

