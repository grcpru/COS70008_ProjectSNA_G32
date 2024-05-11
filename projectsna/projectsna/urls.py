from django.contrib import admin
from django.urls import path
from django.urls import re_path
from django.views.static import serve
from django.conf import settings
from django.conf.urls.static import static
from appg32 import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.landing_page, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('upload/', views.upload_data, name='upload_data'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('success/', views.upload_success, name='upload_success'),
    path('dataset-success/', views.upload_success, name='upload_success'),
    path('explore_datasets/', views.explore_datasets, name='explore_datasets'),
    path('explore_file/<str:file>/', views.explore_file, name='explore_file'),
    path('analyze/', views.analyze_data, name='analyze_data'),
    path('network-graph/', views.network_graph, name='network_graph'),
    path('visualize/', views.network_view, name='network_visualize'),
    path('network-statistics/', views.network_statistics, name='network_statistics'),
    path('analyze-data/', views.analyze_data, name='analyze_data'),  # Your custom ERGM analysis view
    path('plot-results/', views.plot_ergm_results, name='plot_ergm_results'),  # Your custom plot view, if needed
]
""" if settings.DEBUG:
    urlpatterns += [
        re_path(r'^plots/(?P<path>.*)$', serve, {'document_root': settings.PLOT_STORAGE_DIR}),
    ] """
# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
