import cv2
import numpy as np
import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import webbrowser

#load model
@st.cache_resource
def load_emotion_model():
    return load_model() # <- we put the path inside the parenthesis
def recommend_music(emotion):
    #open youtube searching music for the emotion detected
    query = f"{emotion} mood music playlist"
    youtube_url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(youtube_url)
model = load_emotion_model() 
emotion_classes = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

#Define preprocessing function
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (150, 150)) #resize to 150*150
    frame_array = img_to_array(frame_resized) / 255.0 #Normalize pixel values
    frame_array = np.expand_dims(frame_array, axis=0) # Add batch dimensions
# streamlit interface
st.title("Real-Time Emotion Detection with Music Recommendations")
run_webcam = st.checkbox("Start Webcam")
st.text("Check  the box above to start detecting your emotions in real time")
if run_webcam:
    stframe = st.empty() #placeholder for displaying the video frame
    cap = cv2.VideoCapture(0) #0 for the default webcam
    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam")
            break
        #convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #preprocess the frame
        processed_frame = preprocess_frame(frame_rgb) 
        #predict emotion
        predictions = model.predict(processed_frame) 
        detected_emotion  = emotion_classes[np.argmax(predictions)] 
        #Add emotion label to the frame
        cv2.putText(frame, f"Emotion: {detected_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
        #display the frame in streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if st.button("Recommend Music"):
            if detected_emotion:
                recommend_music(detected_emotion)
            else:
                st.warning("No emotion detected yet. Try again.")
        

    cap.release()
else:
    st.text("Webcam is off")
