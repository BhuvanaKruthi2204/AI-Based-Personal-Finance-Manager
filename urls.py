from django.urls import path

from FinanceApp import views

urlpatterns = [path("index.html", views.index, name="index"),
	         path('UserLogin.html', views.UserLogin, name="UserLogin"), 
		     path('Register.html', views.Register, name="Register"), 
		     path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"), 
		     path('LoadDataset', views.LoadDataset, name="LoadDataset"), 
		     path('RegisterAction', views.RegisterAction, name="RegisterAction"), 
		     path('Clustering', views.Clustering, name="Clustering"), 	
		     path('RunLSTM', views.RunLSTM, name="RunLSTM"), 
		     path('Recommend', views.Recommend, name="Recommend"), 
		     path('RecommendAction', views.RecommendAction, name="RecommendAction"), 
		     path('Feedback', views.Feedback, name="Feedback"), 
		     path('FeedbackAction', views.FeedbackAction, name="FeedbackAction"),
		     path('OTPAction', views.OTPAction, name="OTPAction"),
]
