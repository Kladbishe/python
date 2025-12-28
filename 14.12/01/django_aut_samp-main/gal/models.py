from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
    

class MyImage(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="images"
    )
    category = models.ForeignKey(
        Category,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="images"
    )
    desc = models.CharField(max_length=50, null=True, blank=True)
    price = models.DecimalField(max_digits=7, decimal_places=2)  # can store up to 99999.99
    image = models.ImageField(
        # upload_to='images/',  # change path as needed
        null=True,
        blank=True,
        default='images/placeholder.png'
    )
    created_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.desc or 'No description'} - ${self.price}"