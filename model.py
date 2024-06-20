import os
from tensorflow.keras.callbacks import ModelCheckpoint
from helper import *

# Function to get your data
data = get_df()
rangeResults_array = data['rangeResult']
velocities_array = data['velocity']
L_R_array = data['L_R']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(rangeResults_array, velocities_array, test_size=0.2, random_state=42)

# Define the model
model = get_model()

# Define the checkpoint directory and filename
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, 'model_checkpoint.h5')

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',  # Monitor validation loss for improvement
    mode='min',  # Save the model with the minimum validation loss
    save_best_only=True,  # Save only the best model
    verbose=1  # Print out messages when saving the model
)

# Train the model with the checkpoint callback
model = train(model, X_train, y_train, epochs=500, callbacks=[checkpoint_callback])

# Load the best model for testing
model.load_weights(checkpoint_filepath)

# Test the model
test_result = test(model, X_test, y_test)
print(test_result)
