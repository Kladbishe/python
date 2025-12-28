from django.db import models

# Create your models here.
class Product(models.Model):
    desc = models.CharField(max_length=50, null=True, blank=True)
    price = 