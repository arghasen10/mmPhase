import tensorflow as tf
import pandas as pd
import numpy as np
import os
import concurrent.futures
from helper import PointCloudProcessCFG, get_velocity, find_peaks_in_range_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

X_train, X_temp, y_train, y_temp = train_test_split(rangeResultabs, y_TRUE, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(512).repeat(-1).batch(64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(512).repeat(-1).batch(64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.experimental.AUTOTUNE)


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
model.compile(loss=loss_fn,optimizer='adam')
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

history = model.fit(train_dataset, epochs=1000, steps_per_epoch=len(X_train) // 64, validation_data=val_dataset, validation_steps=len(X_val) // 64, callbacks=[checkpoint])

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_loss.png')

model.load_weights('best_model.keras')

y_pred = model.predict(X_test)

mae_metric = tf.keras.metrics.MeanAbsoluteError()
mse_metric = tf.keras.metrics.MeanSquaredError()

mae_metric.update_state(y_test[:, 1], y_pred)
mse_metric.update_state(y_test[:, 1], y_pred)

mae = mae_metric.result().numpy()
mse = mse_metric.result().numpy()

print(f'MAE: {mae}')
print(f'MSE: {mse}')

