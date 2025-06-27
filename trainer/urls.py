from django.urls import path
from . import views

app_name = "trainer"

urlpatterns = [
    path('', views.project_list_view, name='project_list'),
    path('start/<int:project_id>/', views.start_project_view, name='start_project'),
    path('train/', views.train_view, name='train_view'),
    path('train-results/', views.training_result_list_view, name='training_result_list'),
    path("api/update_training/", views.update_training, name="update_training"),
]