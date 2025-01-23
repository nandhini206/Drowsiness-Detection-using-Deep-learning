import cv2
import numpy as np
import streamlit as st
from pygame import mixer
from tensorflow.keras.models import load_model
import os

class DrowsinessDetectionApp:
    def __init__(self):
        # Add robust model loading with error handling
        try:
            # Check if model file exists
            if not os.path.exists('Model.h5'):
                st.error(f"Model file 'Model.h5' not found. Current directory: {os.getcwd()}")
                st.error("Available files: " + str(os.listdir('.')))
                self.model = None
            else:
                try:
                    self.model = load_model('Model.h5')
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    self.model = None

        except Exception as model_load_error:
            st.error(f"Unexpected error during model initialization: {model_load_error}")
            self.model = None

        # Load the cascade classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Initialize audio mixer for alarm sound
        mixer.init()
        self.sound = mixer.Sound('alarm.mp3')  # Ensure the sound file is in the correct path

        # Initialize variables for score and alert system
        self.Score = 0
        self.alert_displayed = False

    def home_page(self):
        st.title('Drowsiness Detection System')
        st.markdown("""
        ## Overview
        Drowsiness Detection is a critical safety technology designed to:
        - Monitor driver alertness in real-time
        - Prevent accidents caused by fatigue
        - Provide timely warnings to maintain road safety

        ### Key Features
        - Advanced eye tracking
        - Machine learning-powered detection
        - Instant audio-visual alerts
        """)
        
        st.image('profile.png', width=None)  # Updated to use width parameter

    def model_implementation_page(self):
        st.title('Drowsiness Detection Implementation')
        
        # Check if model is loaded
        if self.model is None:
            st.warning("Model not loaded. Please check the model file and try again.")
            return
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Placeholder for the final frame
        output_frame = st.empty()
        
        # Checkbox to start/stop detection
        detection_active = st.checkbox('Start Drowsiness Detection', key='start_detection_checkbox', value=True)
        
        # Add a unique key to the stop button
        stop_detection = st.button('Stop Detection', key='stop_detection_button')
        
        if detection_active:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break

                height, width = frame.shape[0:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces and eyes
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

                cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=3)

                for (ex, ey, ew, eh) in eyes:
                    eye = frame[ey:ey+eh, ex:ex+ew]
                    eye = cv2.resize(eye, (80, 80))
                    eye = eye / 255
                    eye = eye.reshape(80, 80, 3)
                    eye = np.expand_dims(eye, axis=0)

                    # Predict eye state (open/closed)
                    prediction = self.model.predict(eye)

                    # Closed eyes detected
                    if prediction[0][0] > 0.30:  
                        cv2.putText(frame, 'closed', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
                        cv2.putText(frame, 'Score: ' + str(self.Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
                        self.Score += 1
                        if self.Score > 15 and not self.alert_displayed:
                            self.sound.play()
                            st.warning("Eyes are closed! Wake up!")
                            self.alert_displayed = True

                    # Open eyes detected
                    elif prediction[0][1] > 0.80:  
                        cv2.putText(frame, 'open', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)      
                        cv2.putText(frame, 'Score: ' + str(self.Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
                        self.Score -= 1
                        if self.Score < 0:
                            self.Score = 0
                        if self.alert_displayed:
                            self.sound.stop()
                            self.alert_displayed = False

                # Update the output frame with the processed frame
                output_frame.image(frame, channels='BGR')  # Removed use_container_width

                # Check if stop button is pressed or detection is inactive
                if stop_detection or not detection_active:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def exit_page(self):
        st.title("Thank You for Using Our Drowsiness Detection System")
        st.balloons()
        st.markdown("---")
    
        st.markdown("""
    
    #### Remember:
    - This drowsiness detection system is designed to assist in preventing accidents caused by driver fatigue
    - Always stay alert while driving, take regular breaks, and avoid driving when feeling tired
    - This system provides a helpful tool but is not a substitute for personal attention to road safety
    
    #### For emergencies:
    - If you feel drowsy while driving, pull over safely and rest
    - Contact emergency services if necessary
    - If you experience severe fatigue, consider traveling with a companion or using alternative transportation
    
    #### Stay Safe on the Road!
    - Get adequate sleep before driving
    - Take regular breaks to avoid fatigue
    - Stay hydrated and avoid long, monotonous drives without rest
    - Be mindful of your alertness levels at all times
    
    #### Thank you for using our system to help ensure safer driving!
    """)

    def run(self):
        # Sidebar configuration
        st.sidebar.title('Drowsiness Detection App')
        st.sidebar.image('camera.png', width=200)  
        st.sidebar.markdown("### About")
        st.sidebar.info("""
    **Developer:** K Dakshata  
                        
    **College:** Sri Vijay Vidyalaya College of Arts and Science  

    This application is built using Streamlit 

""")

        # Page navigation
        app_page = st.sidebar.radio('Navigate', 
            ['Home', 'Model Implementation', 'Exit'])

        if app_page == 'Home':
            self.home_page()
        elif app_page == 'Model Implementation':
            self.model_implementation_page()
        else:
            self.exit_page()

def main():
    app = DrowsinessDetectionApp()
    app.run()

if __name__ == '__main__':
    main()