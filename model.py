from helper import *

df = get_df()
X_train, X_test, y_train, y_test = get_xtrain_ytrain(df, frame_stack=10)
dop_train_s, rp_train_s, noiserp_train_s = preprocess_input(X_train)
model = get_model()
model = train(model, dop_train_s, rp_train_s, noiserp_train_s, y_train, epochs=500)
dop_test_s, rp_test_s, noiserp_test_s = preprocess_input(X_test)
test_result = test(model, dop_test_s, rp_test_s, noiserp_test_s, y_test)