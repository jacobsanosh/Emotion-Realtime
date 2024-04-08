import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model("InceptionV3_Ver1.h5")

# Define the labels for emotions
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

# Create a function to preprocess the input image
# Create a function to preprocess the input image
def preprocess_image(img):
    # Resize the image to match model input size
    resized_img = cv2.resize(img, (139, 139))
    # Convert the resized image to RGB (3 channels)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    # Normalize pixel values to range [0, 1]
    normalized_img = rgb_img.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(normalized_img, axis=0)
    return img_array



# Create a function to detect emotions in real-time using webcam
def detect_emotion():
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Open webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Crop the face region
            roi_gray = gray[y:y+h, x:x+w]
            # Preprocess the face image
            roi_gray = preprocess_image(roi_gray)
            # Predict the emotion
            prediction = model.predict(roi_gray)
            # Get the predicted emotion label
            predicted_emotion = emotions[np.argmax(prediction)]
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Add text with predicted emotion label
            cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the capture
    cap.release()
    # Destroy all windows
    cv2.destroyAllWindows()

# Call the function to detect emotions in real-time
detect_emotion()
