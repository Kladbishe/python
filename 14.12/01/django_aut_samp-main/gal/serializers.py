from rest_framework import serializers
from .models import MyImage

class MyImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyImage
        fields = [
            'id',
            'user',
            'category',
            'desc',
            'price',
            'image',
            'created_time',
        ]
        read_only_fields = ['id', 'created_time','user']

    # Optional: show image URL instead of just file path
    image = serializers.ImageField(max_length=None, use_url=True)
