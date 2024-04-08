import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained emotion classification model
emotion_model = load_model("model_optimal.h5")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the face recognition model
face_recognition_model = load_model("/home/sanosh/Desktop/projects/Emotion_Realtime/keras_model.h5")

# Load the labels for face recognition
class_names = open("labels.txt", "r").readlines()

# Define the labels for emotions
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

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
        # Extract the region of interest (ROI) containing the face for emotion detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        roi_gray_resized = roi_gray_resized / 255.0
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)

        # Predict the emotion label for the ROI image
        emotion_prediction = emotion_model.predict(roi_gray_resized)
        emotion_label = label_dict[np.argmax(emotion_prediction)]

        # Draw a rectangle around the detected face for emotion detection
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Extract the region of interest (ROI) containing the face for face recognition
        roi_color = frame[y:y+h, x:x+w]
        roi_color_resized = cv2.resize(roi_color, (224, 224))

        # Preprocess the ROI image for face recognition
        roi_color_resized = cv2.cvtColor(roi_color_resized, cv2.COLOR_BGR2RGB)
        roi_color_resized = np.expand_dims(roi_color_resized, axis=0)
        roi_color_resized = roi_color_resized / 255.0

        # Predict using the face recognition model
        face_prediction = face_recognition_model.predict(roi_color_resized)
        index = np.argmax(face_prediction)
        class_name = class_names[index]
        confidence_score = face_prediction[0][index]

        # Draw a rectangle around the detected face for face recognition
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the annotated frame in real-time
    cv2.imshow('Real-time Emotion and Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
