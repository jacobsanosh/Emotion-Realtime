import cv2
import face_recognition
import numpy as np
import os
import mediapipe as mp
from keras.models import load_model
from PIL import Image, ImageOps

mp_face_detection = mp.solutions.face_detection
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face recognition models
directory = "/home/sanosh/Desktop/projects/Emotion_Realtime/Students"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(directory):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        image = face_recognition.load_image_file(os.path.join(directory, filename))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

print('Learned encoding for', len(known_face_encodings), 'images.')

# Load emotion detection model
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")
model_facedetection = load_model("/home/sanosh/Desktop/projects/Emotion_Realtime/keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_face_image(face_image, target_size):
    face_image = face_image.convert("RGB").resize(target_size)
    image_array = np.array(face_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                xmin, ymin = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                xmax, ymax = xmin + w, ymin + h
                
                face_image = frame[ymin:ymax, xmin:xmax]
                preprocessed_face = preprocess_face_image(Image.fromarray(face_image), (224, 224))
                data_face = np.expand_dims(preprocessed_face, axis=0)

                # prediction = model_facedetection.predict(data_face)
                # index = np.argmax(prediction)
                # class_name = class_names[index]
                # confidence_score = prediction[0][index]

                roi_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                emotion_prediction = classifier.predict(roi)[0]
                emotion_index = np.argmax(emotion_prediction)
                predicted_emotion = emotion_labels[emotion_index]

                # Perform face recognition
                face_encoding = face_recognition.face_encodings(frame_rgb, [(ymin, xmax, ymax, xmin)])[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    matched_index = matches.index(True)
                    name = known_face_names[matched_index]


                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'{name} {predicted_emotion}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                # cv2.putText(frame, name, (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Detection, Emotion Detection, and Recognition', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
