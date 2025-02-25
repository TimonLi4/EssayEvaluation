from django.urls import path
from .views.views import main_page,upload_file,relevance

urlpatterns = [
    path('',main_page),
    path('stat/',upload_file,name='stat'),
    path('relevance/',relevance,name = 'relevance')
]

