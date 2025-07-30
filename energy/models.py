from django.db import models


# Create your models here.
class ElectricityReading(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    device_id = models.CharField(max_length=50)
    voltage = models.FloatField()
    current = models.FloatField()
    room = models.CharField(max_length=50, null=True, blank=True)
    
class user(models.Model):
    username = models.CharField(max_length=20)
    email = models.EmailField(max_length=20)
    password = models.CharField(max_length=20)
    


    
