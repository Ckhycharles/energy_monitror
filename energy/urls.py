from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from energy import views
import os

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.signup, name='register'),
    path('login/', views.login, name="login"),
    path('dashboard/', views.index, name = "dashboard"),
    path('logout/', views.logout, name='logout'),
    path('dashboard/api/receive-latest/', views.receive_latest_data, name='receive_latest_api'),
    path('dashboard/api/predict/', views.predict_bill, name='predict_bill_api'),
    path('dashboard/export_csv/', views.export_readings_to_csv, name='export_csv'),
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset_form.html'), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
]
