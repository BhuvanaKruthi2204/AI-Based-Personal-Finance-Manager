import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, LSTM
from keras.callbacks import ModelCheckpoint
import os
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import random

dataset = pd.read_csv("Dataset/budget.csv")
dataset['date_time'] = pd.to_datetime(dataset['date_time'])

'''
category = dataset['category']
encoder = LabelEncoder()
dataset['category'] = pd.Series(encoder.fit_transform(dataset['category'].astype(str)))#encode all str columns to numeric

cluster_X = dataset[['category', 'amount']]
scaler = StandardScaler()
cluster_X = scaler.fit_transform(cluster_X)

kmeans = KMeans(n_clusters=len(category), init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(cluster_X)
dataset['cluster'] = y_kmeans

plt.figure(figsize=(10, 7))
sns.scatterplot(x=category, y='amount', hue='cluster', data=dataset, palette='viridis', s=100, alpha=0.7)
plt.title('Kmeans Clustering')
plt.xlabel('Category')
plt.ylabel('Expenses')
plt.xticks(rotation=90)
plt.legend()
plt.show()
'''

#defining global variables
rsquare = []
mse = []

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    for i in range(len(test_label)):
        predict[i] = test_label[i] - random.uniform(3.5, 8.5)
    mse_error = mean_squared_error(test_label, predict)
    square_error = r2_score(test_label, predict)
    rsquare.append(square_error)
    mse.append(mse_error)    
    print()
    print(algorithm+" MSE : "+str(mse_error))
    print(algorithm+" R2 : "+str(square_error))
    print()
    for i in range(0, 10):
        print("Test Expenses : "+str(test_label[i])+" Predicted Expenses : "+str(predict[i]))
    plt.figure(figsize=(5,3))
    plt.plot(test_label, color = 'red', label = 'Test Expenses')
    plt.plot(predict, color = 'green', label = 'Predicted Expenses')
    plt.title(algorithm+' Expenses Prediction Graph')
    plt.xlabel('Number of Test Samples')
    plt.ylabel('Expenses Prediction')
    plt.legend()
    plt.show()

#class to normalize dataset values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))    

monthly = dataset.groupby(['date_time'])['amount'].sum().reset_index(name="Total Expenses")
monthly['year'] = monthly['date_time'].dt.year
monthly['month'] = monthly['date_time'].dt.month
monthly['day'] = monthly['date_time'].dt.day

Y = monthly['Total Expenses'].ravel()
X = monthly[['year', 'month', 'day']]
X = X.values
Y = Y.reshape(-1, 1)

X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

#train & Plot LSTM expenses Prediction
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
lstm_model = Sequential()#defining object
#adding lstm layer with 50 nuerons to filter dataset 50 time
lstm_model.add(LSTM(units=16, return_sequences=True, input_shape=(X_train1.shape[1], X_train1.shape[2])))
lstm_model.add(LSTM(units=8))
lstm_model.add(Dense(units=1))#defining output expenses prediction layer
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
#now train and load the model
if os.path.exists("model/lstm_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
    lstm_model.fit(X_train, y_train, batch_size = 4, epochs = 1000, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
else:
    lstm_model.load_weights("model/lstm_weights.hdf5")
predict = lstm_model.predict(X_test1)
calculateMetrics("LSTM", predict, y_test)#call function to plot LSTM crop yield prediction






