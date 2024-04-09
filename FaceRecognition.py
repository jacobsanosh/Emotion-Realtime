import cv2
import face_recognition
import numpy as np
import os

directory = "/home/sanosh/Desktop/projects/Emotion_Realtime/Students"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(directory):
    if filename.endswith(".jpeg") or filename.endswith(".png")or filename.endswith(".jpg"):  # Filter image files
        image = face_recognition.load_image_file(os.path.join(directory, filename))
        
        # Encode the face in the image
        face_encoding = face_recognition.face_encodings(image)[0]
        
        # Add the face encoding and name to the lists
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension as name

print('Learned encoding for', len(known_face_encodings), 'images.')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the known face's name
        if True in matches:
            matched_index = matches.index(True)
            name = known_face_names[matched_index]

        # Draw a rectangle around the face and label it with the name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
