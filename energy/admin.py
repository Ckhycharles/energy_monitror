from django.contrib import admin
from .models import ElectricityReading

@admin.register(ElectricityReading)
class ElectricityReadingAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'voltage', 'current')
    list_filter = ('timestamp',)
    search_fields = ('timestamp',)
