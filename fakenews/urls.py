from django.urls import path
from . import views

app_name = 'fakenews'

urlpatterns = [
    path('claim/', views.claim, name='claim'),
    path('sentence/', views.sentence, name='sentence'),
    path('tweet/', views.tweet, name='tweet'),
    path('similarity/', views.similarity, name='similarity'),
    # path('savedb/', views.save_data_to_database, name='save_data_to_database'),
    path('', views.home, name='home'),
]