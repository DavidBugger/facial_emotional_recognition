from django.urls import path 
from .views import home_view, detect_emotion, predict_view

urlpatterns = [
    path('', home_view, name='main'),
    path('predict', predict_view, name='predict'),
    path("detect/", detect_emotion, name="detect_emotion"),
]
