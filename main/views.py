from django.shortcuts import render
import cv2
import numpy as np
from django.http import JsonResponse
from tensorflow.keras.models import load_model

# Load pre-trained model
model_path = "model/emotion_model.h5"
model = load_model(model_path)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion Labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def detect_emotion(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        
        # Save the received image for debugging
        with open("received_image.jpg", "wb") as f:
            f.write(image_file.read())
        
        # Reset the file pointer to the beginning
        image_file.seek(0)
        
        # Convert image to OpenCV format
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize image if necessary
        if image.shape[0] > 800 or image.shape[1] > 800:
            image = cv2.resize(image, (640, 480))

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adjust Haar Cascade parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img / 255.0  # Normalize pixel values
            face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
            face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension

            # Predict emotion
            predictions = model.predict(face_img)
            emotion_index = np.argmax(predictions)
            emotion = emotion_labels[emotion_index]

            return JsonResponse({
                "emotion": emotion,
                "face_coordinates": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def home_view(request):
    return render(request,'views/main.html')

def predict_view(request):
    return render(request,'views/predict.html')