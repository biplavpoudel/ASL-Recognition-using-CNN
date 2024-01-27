import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model(r"model\ASL_model\model.h5")

# Defining the classes corresponding to your ASL signs
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Opening a connection to the webcam
cap = cv2.VideoCapture(0)

# Initializing MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Capturing frame-by-frame
    ret, frame = cap.read()

    # Converting the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extracting hand landmarks
            hand_landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])

            # Preprocessing the frame (resize and rescale)
            resized_frame = cv2.resize(frame, (200, 200))
            preprocessed_frame = resized_frame / 255.0  # Assuming the model was trained with values in [0, 1]

            # Expanding dimensions to match the model's expected input shape
            input_frame = np.expand_dims(preprocessed_frame, axis=0)
            print(input_frame.size)

            # Making predictions
            predictions = model.predict(input_frame)
            print("Raw Prediction", predictions)
            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class]


            # Displaying the predicted class and confidence on the frame
            text = f"{class_names[predicted_class]} ({confidence:.2f})"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Drawing hand landmarks on the frame
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Displaying the resulting frame
    cv2.imshow('ASL Prediction with Hand Landmarks', frame)

    # Breaking the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


