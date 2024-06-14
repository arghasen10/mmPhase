from helper import *

df = get_df()
X_train, X_test, y_train, y_test = get_xtrain_ytrain(df, frame_stack=1)
model = get_model()
model = train(model, X_train, y_train, epochs=500)
# dop_test_s, rp_test_s, noiserp_test_s = preprocess_input(X_test)
test_result = test(model, X_test, y_test)

