from helper import *

data = get_df()
rangeResults_array = data['rangeResult']
velocities_array = data['velocity']
L_R_array = data['L_R']
X_train, X_test, y_train, y_test = train_test_split(rangeResults_array, velocities_array, test_size=0.2, random_state=42)
model = get_model()
model = train(model, X_train, y_train, epochs=500)
test_result = test(model, X_test, y_test)