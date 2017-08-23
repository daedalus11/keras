import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("train.csv")

X = training_data_df.drop('Parameter', axis=1).values
Y = training_data_df[['Parameter']].values

# Define the model
model = Sequential()
model.add(Dense(50,input_dim=5,activation='relu'))#everynode connected t everynode. ip = 9
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss = 'mean_squared_error', optimizer='adam')
model.fit(
     X,
     Y,
     epochs=29,
     shuffle=True,
     verbose=2
)

test_data_df= pd.read_csv("test.csv")
X_test = test_data_df.drop('Parameter', axis=1).values
Y_test = test_data_df[['Parameter']].values
test_error_rate = model.evaluate(X_test,Y_test,verbose=0)
print("MSE for test data is: {}".format(test_error_rate))

X = pd.read_csv("test.csv").values
prediction = model.predict(X)
prediction = prediction[0][0]
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968
print("Prediction - ${}".format(prediction))
