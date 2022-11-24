### Exp No: 05
### Date: 16.10.2022
# <p align="center">Stock Price Prediction</p>

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## PROBLEM STATEMENT AND DATASET
We aim to build a RNN model to predict the stock prices of Google using the dataset provided. The dataset has many features, but we will be predicting the "Open" feauture alone. We will be using a sequence of 60 readings to predict the 61st reading. <br>Note: These parameters can be changed as per requirements.

## RECURRENT NEURAL NETWORK MODEL

![WhatsApp Image 2022-10-13 at 18 37 08](https://user-images.githubusercontent.com/77089743/195605336-3d92175d-0ffe-4ae2-9341-eb9d3b5e36d1.jpeg)


## DESIGN STEPS

### Step 1:
Read the csv file and create the Data frame using pandas.
### Step 2:
Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.
### Step 3:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train. 
### Step 4:
Create a model with the desired number of nuerons and one output neuron.
### Step 5: 
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.
### Step 6:
Make Predictions and plot the graph with the Actual and Predicted values.

## PROGRAM

```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

dataset_train.tail()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

plt.plot(np.arange(0,1259),train_set)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()
## Write your code here
model.add(layers.SimpleRNN(50,input_shape=(length, n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test=[]
for i in range(50,1384):
  X_test.append(inputs_scaled[i-50:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(50,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://user-images.githubusercontent.com/77089743/195605002-09873d8c-db15-4959-8e11-c5dc075f7d0c.png)

<br>
<br>
<br>
<br>
<br>
<br>

### Mean Square Error

![image](https://user-images.githubusercontent.com/77089743/195605127-73cd0494-c13c-4b19-9373-c728b1e596e4.png)

## RESULT
Thus, we have successfully created a Simple RNN model for Stock Price Prediction.

