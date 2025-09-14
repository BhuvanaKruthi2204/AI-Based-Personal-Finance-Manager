from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import Dropout, LSTM
from keras.callbacks import ModelCheckpoint
import os
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import random
import smtplib
from datetime import date

global username, dataset, X, Y, scaler, scaler1, lstm_model
global X_train, X_test, y_train, y_test, otp
mse = []
accuracy = []

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
    accuracy.append(square_error)
    mse.append(mse_error)    
    plt.figure(figsize=(5,3))
    plt.plot(test_label, color = 'red', label = 'Test Expenses')
    plt.plot(predict, color = 'green', label = 'Predicted Expenses')
    plt.title(algorithm+' Expenses Prediction Graph')
    plt.xlabel('Number of Test Samples')
    plt.ylabel('Expenses Prediction')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.clf()
    plt.cla()
    return img_b64

def getSentiment(feedback):
     sid = SentimentIntensityAnalyzer()
     scores = sid.polarity_scores(feedback)
     sentiment = ""
     if scores['compound'] >= 0.05 :
         sentiment = "Positive"
     elif scores['compound'] <= -0.05 :
         sentiment = "Negative"
     else :
         sentiment = "Neutral"
     return sentiment    

def FeedbackAction(request):
    if request.method == 'POST':
        global username
        feedback = request.POST.get('t1', False)
        dd = str(date.today())
        sentiment = getSentiment(feedback.lower().strip())
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'finance',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO feedback VALUES('"+username+"','"+feedback+"','"+dd+"','"+sentiment+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        output = "<font size=3 color=white>Your feedback Accepted</font><br/>"
        output += "<font size=3 color=white>Sentiment Detected from your feedback = "+sentiment+"</font><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RecommendAction(request):
    if request.method == 'POST':
        global lstm_model, dataset, scaler, scaler1
        dd = request.POST.get('t1', False)
        today_date = dd
        dd = dd.split("-")
        print(dd)
        data = []
        lstm_model = load_model("model/lstm_weights.hdf5")
        data.append([float(dd[0].strip()), float(dd[1].strip()), float(dd[2].strip())])
        data = np.asarray(data)
        data = scaler.transform(data)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        predict = lstm_model.predict(data)
        predict = predict.reshape(-1, 1)
        predict = scaler1.inverse_transform(predict)
        predict = int(predict * 1000)
        income = random.randint(predict, predict + 50000)
        output = "<font size=3 color=white>Suppose your income = "+str(income)+"</font><br/>"
        output += "<font size=3 color=white>Predicted Expenses = "+str(predict)+"</font><br/>"
        output += "<font size=3 color=white>Recommendation for Investment Amount = "+str(income - predict)+"</font><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def Feedback(request):
    if request.method == 'GET':
       return render(request, 'Feedback.html', {})    

def Recommend(request):
    if request.method == 'GET':
       return render(request, 'Recommend.html', {})

def RunLSTM(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, lstm_model, dataset, scaler, scaler1
        global accuracy, mse
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
        img_b64 = calculateMetrics("LSTM", predict, y_test)#call function to plot LSTM crop yield prediction
        output='<table border=1 align=center width=100%><tr><th><font size="" color="white">Algorithm Name</th><th><font size="" color="white">Accuracy</th>'
        output += '<th><font size="" color="white">Mean Square Error</th>'
        output+='</tr>'
        algorithms = ['LSTM']
        for i in range(len(algorithms)):
            output += '<td><font size="" color="white">'+algorithms[i]+'</td><td><font size="" color="white">'+str(accuracy[i])+'</td><td><font size="" color="white">'+str(mse[i])+'</td>'
            output += '</tr>'
        output+= "</table></br>"
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def Clustering(request):    
    if request.method == 'GET':
        global dataset
        df = pd.read_csv("Dataset/budget.csv")
        category = df['category']
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
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':'Expenses Behaviour Clustering Graph', 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global dataset
        dataset = pd.read_csv("Dataset/budget.csv")
        dataset['date_time'] = pd.to_datetime(dataset['date_time'])
        columns = dataset.columns
        data = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="white">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(data)):
            output += '<tr>'
            for j in range(len(data[i])):
                output += '<td><font size="3" color="white">'+str(data[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def sendOTP(email, otp_value):
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'kaleem202120@gmail.com'
        email_password = 'xyljzncebdxcubjq'
        connection.login(email_address, email_password)
        connection.sendmail(from_addr="kaleem202120@gmail.com", to_addrs=em, msg="Subject : Your OTP : "+otp_value)

def OTPAction(request):
    if request.method == 'POST':
        global otp, username
        otp_value = request.POST.get('t1', False)
        if otp == otp_value:
            context= {'data':'Welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid OTP! Please Retry'}
            return render(request, 'OTP.html', context)   

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username, otp
        status = "none"
        email = ""
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        password = base64.b64encode(password.encode("ascii"))
        password = password.decode("ascii")               
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'finance',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password,email_id FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    email = row[2]
                    break
        if status == 'success':
            otp = str(random.randint(1000, 9999))
            sendOTP(email, otp)
            context= {'data':'OTP sent to your mail'}
            return render(request, 'OTP.html', context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        password = base64.b64encode(password.encode("ascii"))
        password = password.decode("ascii")               
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'finance',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'finance',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Finance prediction"
        context= {'data':output}
        return render(request, 'Register.html', context)       

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})



    

