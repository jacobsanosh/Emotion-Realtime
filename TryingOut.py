import cv2
import mediapipe as mp
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the face detection model
model_facedetection = load_model("/home/sanosh/Desktop/projects/Emotion_Realtime/keras_model.h5", compile=False)

# Load the emotion recognition model
model_emotion = load_model("model_optimal.h5", compile=False)

# Load the emotion labels
class_names = open("labels.txt", "r").readlines()

# Define the label dictionary for emotions
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def preprocess_face_image(face_image, target_size):
    # Convert image to grayscale and resize to target size
    face_image = face_image.convert("L").resize(target_size)

    # Convert image to array
    image_array = np.array(face_image)

    # Normalize pixel values for the emotion recognition model
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Expand dimensions to match expected input shape (add channel dimension)
    normalized_image_array = np.expand_dims(normalized_image_array, axis=-1)

    return normalized_image_array

def preprocess_face_image2(face_image, target_size):
    # Convert image to RGB and resize to target size
    face_image = face_image.convert("RGB").resize(target_size)

    # Convert image to array
    image_array = np.array(face_image)

    # Normalize pixel values for the face detection model
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    return normalized_image_array


cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_detection.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                xmin, ymin = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                xmax, ymax = xmin + w, ymin + h
                
                face_image = image[ymin:ymax, xmin:xmax]
                
                # Preprocess face image for face detection model
                preprocessed_face = preprocess_face_image2(Image.fromarray(face_image), (224, 224))
                data_face_detection = np.expand_dims(preprocessed_face, axis=0)

                # Predict using the face detection model
                prediction = model_facedetection.predict(data_face_detection)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                
                # Preprocess face image for emotion recognition model
                preprocessed_face_emotion = preprocess_face_image(Image.fromarray(face_image), (48, 48))
                data_emotion_recognition = np.expand_dims(preprocessed_face_emotion, axis=0)
                
                # Predict using the emotion recognition model
                emotion_prediction = model_emotion.predict(data_emotion_recognition)
                emotion_index = np.argmax(emotion_prediction)
                emotion_label = label_dict[emotion_index]

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f'{class_name[2:][:-1]} {emotion_label}', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('Face Detection and Classification', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
