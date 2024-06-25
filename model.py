from helper import get_df, get_model, train, test
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os

def save_model(model, epoch, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.h5")
    model.save(model_path)
    print(f"Model saved at {model_path}")

def train_batchwise(model, X_train, y_train, batch_size, epochs, save_interval=10):
    num_samples = len(X_train)
    num_batches = num_samples // batch_size
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Update the model using the batch
            model = train(model, X_batch, y_batch, epochs=5)
            
        # Handle the last batch if the number of samples is not perfectly divisible by batch_size
        if num_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            X_batch = X_train[start_idx:]
            y_batch = y_train[start_idx:]
            model = train(model, X_batch, y_batch, epochs=1)
        
        # Save the model at the specified interval
        if (epoch + 1) % save_interval == 0:
            save_model(model, epoch)
    
    return model

# Load the data
data = get_df()
rangeResults_array = data['rangeResult']
velocities_array = data['velocity']

print("----------------------------DATA LOADED----------------------------")
# Normalize the range results array if mse is 1
mse = 1
if mse == 1:
    rangeResults_array = np.abs(rangeResults_array)
    rangeResults_array = np.sum(rangeResults_array, axis=(1,2))
    rangeResults_array = np.expand_dims(rangeResults_array, -1)
    rangeResults_array = (rangeResults_array - np.min(rangeResults_array)) / (np.max(rangeResults_array) - np.min(rangeResults_array))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(rangeResults_array, velocities_array, test_size=0.2, random_state=42)

X_train = tf.convert_to_tensor(X_train, dtype='float64')
X_test = tf.convert_to_tensor(X_test, dtype='float64')
y_train = tf.convert_to_tensor(y_train, dtype='float64')
y_test = tf.convert_to_tensor(y_test, dtype='float64')

# Get the model
model = get_model()
print("----------------------------MODEL LOADED----------------------------")
# Train the model batch-wise
batch_size = 32
epochs = 500
save_interval = 10  # Save the model every 10 epochs
model = train_batchwise(model, X_train, y_train, batch_size, epochs, save_interval)

# Test the model
test_result = test(model, X_test, y_test)
print(test_result) 
