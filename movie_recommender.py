import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model

def detect_face(img):
    coord = haar.detectMultiScale(img)
    
    return coord

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import pandas as pd

# Function to reassemble the model parts
def reassemble_model(parts, output_file):
    with open(output_file, 'wb') as f_out:
        for part in parts:
            with open(part, 'rb') as f_in:
                f_out.write(f_in.read())
    print(f"Model reassembled to {output_file}")

# List of the parts
parts = ['emotion_recognition_model_part_aa', 'emotion_recognition_model_part_ab', 'emotion_recognition_model_part_ac']

# Reassemble the model
reassemble_model(parts, 'emotion_recognition_model_reassembled.h5')

# Load the reassembled model
model_tuned = load_model("emotion_recognition_model_reassembled.h5")
# Load the model and face detector
haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model_tuned = load_model("emotion_recognition_model.h5")

# Load movie data
try:
    movie_data = pd.read_csv('movies.csv')
    print("Movie data loaded successfully.")
except Exception as e:
    print(f"Error loading movie data: {e}")
    movie_data = None  # Handle missing movie data gracefully

# Emotion classes
classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def get_random_movie(emotion):
    if movie_data is None:
        return "Movie data unavailable."
    filtered_movies = movie_data[movie_data['Emotion'] == emotion]
    if not filtered_movies.empty:
        random_movie = filtered_movies.sample(n=1)
        return f"Recommended Movie: {random_movie['Movie Title'].values[0]} ({random_movie['Genre'].values[0]})"
    else:
        return "No recommendations available for this emotion."

def detect_face(img):
    coord = haar.detectMultiScale(img)
    print("Faces detected:", coord)
    return coord

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam opened successfully.")

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coords = detect_face(gray_frame)

    for x, y, w, h in coords:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        face_crop = gray_frame[y:y + h, x:x + w]
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = np.expand_dims(face_crop, axis=-1)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model_tuned.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"Emotion Detected: {label}")
        print(get_random_movie(label))

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

