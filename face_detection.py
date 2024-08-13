import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import time

# Capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

# Load the model and load the weights
face_exp_model = model_from_json(open("facial_expression_model_structure.json","r",encoding="utf-8").read())
face_exp_model.load_weights('facial_expression_model_weights.h5')

# Declare the emotions label
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Initialize variables to hold counts and timers
neutral_count = 0
happy_count = 0
sad_count = 0
start_time = time.time()

# Create a separate window to display the face
cv2.namedWindow("Face", cv2.WINDOW_NORMAL)

# Loop through every frame in the video
while True:
    # Get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    
    # Resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    
    # Detect all faces in the image
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')
    
    # Loop through the face locations
    for current_face_location in all_face_locations:
        # Extract the face from the frame
        top_pos, right_pos, bottom_pos, left_pos = [i*4 for i in current_face_location]
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        # Preprocess input, convert it to an image like as the data in dataset
        current_face_image_gray = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
        current_face_image_gray = cv2.resize(current_face_image_gray, (48, 48))
        img_pixels = image.img_to_array(current_face_image_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 

        # Do prediction using model
        exp_predictions = face_exp_model.predict(img_pixels) 
        max_index = np.argmax(exp_predictions[0])
        emotion_label = emotions_label[max_index]
        
        # Count occurrences of emotions
        if emotion_label == 'neutral':
            neutral_count += 1
        elif emotion_label == 'happy':
            happy_count += 1
        elif emotion_label == 'sad':
            sad_count += 1
    
    # Check if 5 minutes have passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= 5:  # 300 seconds = 5 minutes
        break

    # Display the face in a separate window
    cv2.imshow("Face", current_face_image)

    # Wait for a small amount of time to allow the window to refresh
    cv2.waitKey(1)

# Print the counts of each emotion
print("Neutral count:", neutral_count)
print("Happy count:", happy_count)
print("Sad count:", sad_count)

# Release the video stream and close all OpenCV windows
webcam_video_stream.release()
cv2.destroyAllWindows()


