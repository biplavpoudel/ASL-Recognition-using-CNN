import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the saved hand classification model
# model = tf.keras.models.load_model(r"")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            hand_landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])

            # Make predictions using the hand classification model
            # prediction = model.predict(np.expand_dims(hand_landmarks_np.flatten(), axis=0))

            # Get the predicted class
            # predicted_class = np.argmax(prediction)

            # Display the predicted class on the frame
            # cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw hand landmarks on the frame
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Landmark Detection and Classification', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
