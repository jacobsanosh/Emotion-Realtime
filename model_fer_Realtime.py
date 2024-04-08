import cv2
import numpy as np
from keras.models import load_model

# Load the trained emotion classification model
model = load_model("model_optimal.h5")

# Define the labels for emotions
label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Preprocess the ROI image to match the input size and format expected by the model
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        roi_gray_resized = roi_gray_resized / 255.0
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)
        
        # Predict the emotion label for the ROI image
        prediction = model.predict(roi_gray_resized)
        emotion_label = label_dict[np.argmax(prediction)]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the predicted emotion label above the rectangle
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the annotated frame in real-time
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
