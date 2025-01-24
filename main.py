from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Initialize FastAPI app
app = FastAPI()

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().splitlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip and process frame
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        height, width, _ = frame.shape
        className = ""

        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                landmarks = []
                x_coords = []
                y_coords = []

                for lm in handslms.landmark:
                    lmx = int(lm.x * width)
                    lmy = int(lm.y * height)
                    landmarks.append([lmx, lmy])
                    x_coords.append(lmx)
                    y_coords.append(lmy)

                # Calculate bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

                # Draw landmarks
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Reshape landmarks to match model input (21, 2)
                landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                
                # Predict gesture
                if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
                    prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
                    classID = np.argmax(prediction)
                    className = classNames[classID]

        # Show prediction on frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
def index():
    return {"message": "Welcome to the Hand Gesture Recognition API"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# Close the webcam when server shuts down
@app.on_event("shutdown")
def shutdown_event():
    cap.release()

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
