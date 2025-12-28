from rest_framework import serializers
from .models import Product, Category

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'desc', 'price', 'category', 'image', 'createdTime']
        read_only_fields = ['id', 'createdTime']