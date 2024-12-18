import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import gradio as gr
import keras
import requests
import random

# Function to fetch random movies by genre with poster images
def fetch_random_movies_by_genre(genre_id=10749, max_certification='PG-13'):
    try:
        random_page = random.randint(1, 50)  # Randomize page for diversity
        url = f"https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": "b060bfcd031f0398dcd2bcb5f56d40a1",
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "page": random_page,
            "certification_country": "US",  # Use country-specific certifications (e.g., US)
            "certification.lte": max_certification,  # Restrict to certification <= max_certification
        }
        
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return f"Error: Unable to fetch data from TMDB. Status Code: {response.status_code}"

        results = response.json().get("results", [])
        if not results:
            return "No movies found matching the criteria."

        # Prepare movie data
        movies = []
        for movie in results[:5]:  # Limit to 5 movies
            movies.append({
                "title": movie.get("title", "Unknown"),
                "image_url": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get("poster_path") else "https://via.placeholder.com/500",
                "rating": movie.get("vote_average", "N/A"),
                "year": movie.get("release_date", "").split("-")[0] if movie.get("release_date") else "N/A",
                "summary": movie.get("overview", "No summary available."),
            })

        return movies

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to generate a movie recommendation based on the most frequent emotion
def recommend_movies_based_on_emotion(emotion_summary):
    if not emotion_summary:
        return "No emotions detected yet. Process a video first."

    # Parse the summary text to find the most frequent emotion
    try:
        emotion_counts = {line.split(": ")[0]: int(line.split(": ")[1]) for line in emotion_summary.split("\n") if ": " in line}
        most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
    except Exception:
        return "Invalid emotion summary format. Please ensure the summary is correctly generated."

    # Map emotions to TMDB genre IDs
    genre_map = {
        "Angry": 28,    # Action
        "Fear": 27,     # Horror
        "Happy": 35,    # Comedy
        "Neutral": 18,  # Drama
        "Sad": 10749,   # Romance
        "Surprise": 9648  # Mystery
    }

    genre_id = genre_map.get(most_frequent_emotion, 35)  # Default to Comedy

    # Fetch movies with posters
    movies = fetch_random_movies_by_genre(genre_id)

    if isinstance(movies, str):  # Error handling
        return movies

    # Format movie data with posters
    movie_recommendations = f"<b>Top recommendations for '{most_frequent_emotion}' emotion:</b><br><br>"
    for movie in movies:
        movie_recommendations += f"<b>{movie['title']}</b> ({movie['year']}) - Rating: {movie['rating']}<br>{movie['summary']}<br>"
        movie_recommendations += f"<img src='{movie['image_url']}' alt='{movie['title']}' width='200'><br><br>"  # Add movie poster image

        image_html = f"<br><img src='{'blue_short.svg'}' alt='Image' width='300'><br><br><b>{'Credits'}</b>"    
        
    return movie_recommendations


# Video Inference Function with Emotion Recognition Model
def vid_inf(vid):
    emotion_model = keras.models.load_model('emotion_recognition_model_v2.h5')  # Adjust the path if needed

    cap = cv2.VideoCapture(vid)  # Start capturing video from the file
    frame_width = int(cap.get(3))  # Get video width
    frame_height = int(cap.get(4))  # Get video height
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_size = (frame_width, frame_height)  # Determine the size of video frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec
    output_video = "output_emotion_recognition.mp4"  # Output file name

    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)  # Create VideoWriter object

    # Define emotion labels
    emotion_labels = ['Angry','Fear','Happy','Neutral','Sad','Surprise']

    # Initialize frame counts for each emotion
    emotion_counts = {label: 0 for label in emotion_labels}

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Iterate through each detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box
                roi_gray = gray_frame[y:y + h, x:x + w]  # Extract face region
                roi_gray = cv2.resize(roi_gray, (150, 150), interpolation=cv2.INTER_AREA)  # Resize to expected input size

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype("float") / 255.0  # Normalize pixel values
                    roi = img_to_array(roi)  # Convert image to array
                    roi = np.expand_dims(roi, axis=0)  # Add batch dimension

                    # Predict emotion
                    preds = emotion_model.predict(roi)[0]
                    detected_emotion = emotion_labels[preds.argmax()]  # Get the emotion with the highest probability

                    # Increment the emotion count
                    emotion_counts[detected_emotion] += 1

                    # Draw the detected emotion label
                    cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)  # Write the frame to the output video file
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None, None  # Yield the processed frame
        else:
            break

    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Generate summary text for emotion counts
    emotion_summary = "\n".join([f"{emotion}: {count}" for emotion, count in emotion_counts.items()])

    # Generate movie recommendations
    movie_recommendations = recommend_movies_based_on_emotion(emotion_summary)

    # Final yield includes processed video, emotion summary, and movie recommendations
    yield None, output_video, emotion_summary, movie_recommendations


# Gradio Interface for Emotion Recognition and Movie Recommendation
input_video = gr.Video(sources=None, label="Input Video")
output_frame = gr.Image(type="numpy", label="Output Frames")
output_video_file = gr.Video(label="Output Video")
output_summary = gr.Textbox(label="Emotion Summary")
output_movies = gr.HTML(label="Movie Recommendations")  # Use gr.HTML

# Main interface for emotion recognition and movie recommendation
interface_video = gr.Interface(
    fn=vid_inf,
    inputs=[input_video],
    outputs=[output_frame, output_video_file, output_summary, output_movies],
    title="Emotion Recognition in Video",
    description="Upload your video and see the emotion recognition results along with a summary of detected emotions and movie recommendations!",
    examples=[["sample/video_1.mp4"], ["sample/person.mp4"]],
    cache_examples=False,
)

# Function to redo recommendations based on emotion summary
def redo_recommendations(emotion_summary):
    return recommend_movies_based_on_emotion(emotion_summary)

# Redo recommendations button
def create_redo_button(emotion_summary):
    return gr.Button("Redo Recommendations").click(fn=redo_recommendations, inputs=emotion_summary, outputs=output_movies)

# Interface setup with button for redo
with gr.Blocks() as app:
    interface_video.render()  # Render video interface first
    gr.Button("Redo Recommendations").click(redo_recommendations, inputs=[output_summary], outputs=[output_movies])  # Button to trigger redo recommendations

# Launch both interfaces
app.launch(share=True)

