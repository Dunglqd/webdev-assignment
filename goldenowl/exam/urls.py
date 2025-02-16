from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('search/', views.search_scores_view, name='search_scores'),
    path('report/', views.report_view, name='report'),
    path('top-group-a/', views.top_group_a_view, name='top_group_a'),
]


