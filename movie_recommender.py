import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import csv
import random

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Map numeric predictions to emotion labels
emotion_labels = ['angry', 'happy', 'sad', 'surprised', 'neutral', 'fear']

def load_movies(file_path):
    """Loads movies from the CSV file and organizes them by emotion."""
    movies_by_emotion = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            emotion = row['Emotion'].strip().lower()
            if emotion not in movies_by_emotion:
                movies_by_emotion[emotion] = []
            movies_by_emotion[emotion].append((row['Movie Title'], row['Genre']))
    return movies_by_emotion

def recommend_movie(emotion, movies_by_emotion):
    """Recommends a random movie based on the emotion."""
    if emotion in movies_by_emotion:
        movie, genre = random.choice(movies_by_emotion[emotion])
        return f"We recommend you watch '{movie}' ({genre})."
    else:
        return "Sorry, we couldn't find any movies for that emotion."

def predict_emotion(image_path):
    """Predict the emotion from the input image."""
    img = load_img(image_path, target_size=(150, 150), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

def main():
    # Load movies CSV
    movies_file_path = 'movies.csv'
    movies_by_emotion = load_movies(movies_file_path)
    
    # Input: Path to the image
    image_path = input("Enter the path to the emotion image: ").strip()
    
    # Predict the emotion
    detected_emotion = predict_emotion(image_path)
    print(f"Detected Emotion: {detected_emotion}")
    
    # Recommend a movie based on the detected emotion
    recommendation = recommend_movie(detected_emotion, movies_by_emotion)
    print(recommendation)

if __name__ == "__main__":
    main()
