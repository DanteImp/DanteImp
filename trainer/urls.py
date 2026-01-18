from django.urls import path
from . import views

app_name = "trainer"

urlpatterns = [
    path("", views.project_list, name="project_list"),
    path("projects/create/", views.project_create, name="project_create"),
    path("projects/<int:project_id>/train/", views.train_start, name="train_start"),
    path("train/", views.train_view, name="train_view"),
    path("train-results/", views.training_result_list, name="training_result_list"),
    path("api/update_training/", views.update_training, name="update_training"),
]
