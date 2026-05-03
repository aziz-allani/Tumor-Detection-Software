from django.urls import path
from . import views



#urlpatterns = [
#    path('aa/', views.upload_image, name='upload_image'),
#]

urlpatterns = [
    path('', views.index_view, name='index_view'),
    path('home/', views.homepage_view, name='homepage'),  # Route for the homepage
    path('aa/', views.load_data, name='load_data'),
    path('plot/', views.plot_data_distributions, name='plot_data_distributions'),
    path('show-images/', views.show_images_view, name='show_images'),
    path('pre/', views.preprocess_view, name='preprocess'),
    path('plot-augmented-images/', views.plot_augmented_images_view, name='plot_augmented_images'),
    path('preprocess-dataset/', views.preprocess_dataset_view, name='preprocess_dataset'),
    path('build-model/', views.build_model_view, name='build_model'),
    path('train-model/', views.train_model_view, name='train_model'),
    path('evaluate-model/', views.evaluate_model_view, name='evaluate_model'),
    path('plot-history/', views.plot_training_history_view, name='plot_history'),
    path('upload-image/', views.upload_image_view, name='upload_image'),
    # Add more URL patterns here
]




