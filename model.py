from helper import *

data = get_df()
rangeResults_array = data['rangeResult']
velocities_array = data['velocity']
L_R_array = data['L_R']
def get_batch(size=64):
    rand_idx=np.random.randint(0,L_R_array.shape[0],size)
    return rangeResults_array[rand_idx],L_R_array[rand_idx],velocities_array[rand_idx]
X_train, X_test, y_train, y_test = train_test_split(rangeResults_array, velocities_array, test_size=0.2, random_state=42)
model = get_model()
model = train(model, X_train, y_train, L_R_array, epochs=500)
# model = traincnn(model, X_train, y_train, epochs=500)
test_result = test(model, X_test, y_test)