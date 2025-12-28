from django.shortcuts import render
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from .models import MyImage
from .serializers import MyImageSerializer

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def myimage_list_create(request):
    if request.method == 'GET':
        # images = MyImage.objects.all()
        images = MyImage.objects.filter(user=request.user)
        serializer = MyImageSerializer(images, many=True, context={'request': request})
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = MyImageSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)













@api_view(['GET'])
def index(req):
    return Response('index')

# Register - new user
# login - DONE
# add image to a user
# get images per user



@api_view(['POST'])
def register_user(request):
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')

    if not username or not password:
        return Response({"error": "username and password are required"}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username=username).exists():
        return Response({"error": "username already exists"}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(username=username, email=email, password=password)

    return Response({"success": f"user {user.username} created"}, status=status.HTTP_201_CREATED)
