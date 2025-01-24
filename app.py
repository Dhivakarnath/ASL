# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
# from fastapi.responses import StreamingResponse
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize MediaPipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # Dictionary to keep track of users' WebSocket connections
# user_sockets = {}

# # WebSocket endpoint for individual video stream
# @app.websocket("/video_feed/{user_id}")
# async def video_feed(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket
#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Form
# from fastapi.responses import StreamingResponse
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize MediaPipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # Dictionary to keep track of users' WebSocket connections
# user_sockets = {}

# # API endpoint to connect user by user ID
# @app.post("/connect_user/")
# async def connect_user(user_id: str = Form(...)):
#     # This will connect the user and start the webcam feed for the user
#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     if not cap.isOpened():
#         raise HTTPException(status_code=400, detail="Camera not accessible")

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)

#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             # Send the frame to the user (this would typically be sent via WebSocket in a production environment)
#             # In this case, for the sake of streaming, let's simulate a response using StreamingResponse
#             return StreamingResponse(io.BytesIO(frame), media_type="image/jpeg")

#             # Adjust to control frame rate (e.g., 30 FPS)
#             await asyncio.sleep(0.03)
#     except Exception as e:
#         cap.release()
#         raise HTTPException(status_code=500, detail=str(e))

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, Form, Request, HTTPException, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize MediaPipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # Initialize template rendering engine
# templates = Jinja2Templates(directory="templates")

# # This will keep track of users' WebSocket connections
# user_sockets = {}

# # Route for connecting a user and getting their name
# @app.get("/connect_user/", response_class=HTMLResponse)
# async def connect_user_form(request: Request):
#     # Render the form to get the user's name
#     return templates.TemplateResponse("connect_user.html", {"request": request})

# # Route to handle the form submission and redirect to the video feed page
# @app.post("/connect_user/")
# async def connect_user(user_name: str = Form(...)):
#     # Generate a unique user ID (you can make this more sophisticated if needed)
#     user_id = user_name  # For simplicity, using the name as the user_id
    
#     # Redirect to the video feed page for that user
#     return RedirectResponse(url=f"/video_feed/{user_id}")

# # Video feed route where user can access the webcam feed
# @app.get("/video_feed/{user_id}", response_class=HTMLResponse)
# async def video_feed(request: Request, user_id: str):
#     # This page shows the user's video feed
#     return templates.TemplateResponse("video_feed.html", {"request": request, "user_id": user_id})

# # WebSocket endpoint to stream video
# @app.websocket("/video_stream/{user_id}")
# async def video_stream(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket

#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Form, Request, HTTPException, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize template rendering engine
# templates = Jinja2Templates(directory="templates")

# # Initialize MediaPipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # This will keep track of users' WebSocket connections
# user_sockets = {}

# # Route for connecting a user and getting their name
# @app.get("/connect_user/", response_class=HTMLResponse)
# async def connect_user_form(request: Request):
#     # Render the form to get the user's name
#     return templates.TemplateResponse("connect_user.html", {"request": request})

# # Route to handle the form submission and redirect to the video feed page
# @app.post("/connect_user/")
# async def connect_user(user_name: str = Form(...)):
#     # Generate a unique user ID (you can make this more sophisticated if needed)
#     user_id = user_name  # For simplicity, using the name as the user_id
    
#     # Redirect to the video feed page for that user
#     return RedirectResponse(url=f"/video_feed/{user_id}", status_code=303)  # 303 See Other to enforce GET method

# # Route for video feed page where the user will be redirected after name submission
# @app.get("/video_feed/{user_id}", response_class=HTMLResponse)
# async def video_feed(request: Request, user_id: str):
#     # This page shows the user's video feed
#     return templates.TemplateResponse("video_feed.html", {"request": request, "user_id": user_id})

# # WebSocket endpoint to stream video
# @app.websocket("/video_stream/{user_id}")
# async def video_stream(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket

#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize template rendering engine
# templates = Jinja2Templates(directory="templates")

# # Initialize MediaPipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # This will keep track of users' WebSocket connections
# user_sockets = {}

# # Route for connecting a user and getting their name
# @app.get("/connect_user/", response_class=HTMLResponse)
# async def connect_user_form(request: Request):
#     # Render the form to get the user's name
#     return templates.TemplateResponse("connect_user.html", {"request": request})

# # Route to handle the form submission and redirect to the video feed page
# @app.post("/connect_user/")
# async def connect_user(user_name: str = Form(...)):
#     # Generate a unique user ID (you can make this more sophisticated if needed)
#     user_id = user_name  # For simplicity, using the name as the user_id
    
#     # Redirect to the video feed page for that user
#     return RedirectResponse(url=f"/video_feed/{user_id}", status_code=303)  # 303 See Other to enforce GET method

# # Route for video feed page where the user will be redirected after name submission
# @app.get("/video_feed/{user_id}", response_class=HTMLResponse)
# async def video_feed(request: Request, user_id: str):
#     # This page shows the user's video feed
#     return templates.TemplateResponse("video_feed.html", {"request": request, "user_id": user_id})

# # WebSocket endpoint to stream video
# @app.websocket("/video_stream/{user_id}")
# async def video_stream(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket

#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     if not cap.isOpened():
#         await websocket.send_text("Failed to open the camera.")
#         return

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize template rendering engine
# templates = Jinja2Templates(directory="templates")

# # Initialize MediaPipe for hand gesture recognition
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # This will keep track of users' WebSocket connections
# user_sockets = {}

# # Route for connecting a user and getting their name
# @app.get("/connect_user/", response_class=HTMLResponse)
# async def connect_user_form(request: Request):
#     # Render the form to get the user's name
#     return templates.TemplateResponse("connect_user.html", {"request": request})

# # Route to handle the form submission and redirect to the video feed page
# @app.post("/connect_user/")
# async def connect_user(user_name: str = Form(...)):
#     # Generate a unique user ID (you can make this more sophisticated if needed)
#     user_id = user_name  # For simplicity, using the name as the user_id
    
#     # Redirect to the video feed page for that user
#     return RedirectResponse(url=f"/video_stream/{user_id}", status_code=303)  # 303 See Other to enforce GET method

# # WebSocket endpoint to stream video
# @app.websocket("/video_stream/{user_id}")
# async def video_stream(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket

#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     if not cap.isOpened():
#         await websocket.send_text("Failed to open the camera.")
#         return

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io
# import uuid  # For generating dynamic user IDs

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize template rendering engine
# templates = Jinja2Templates(directory="templates")

# # Initialize MediaPipe for hand gesture recognition
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # This will keep track of users' WebSocket connections
# user_sockets = {}

# # Route for connecting a user and getting their name
# @app.get("/connect_user/", response_class=HTMLResponse)
# async def connect_user_form(request: Request):
#     # Render the form to get the user's name
#     return templates.TemplateResponse("connect_user.html", {"request": request})

# # Route to handle the form submission and redirect to the video feed page with dynamic user ID
# @app.post("/connect_user/")
# async def connect_user(user_name: str = Form(...)):
#     # Generate a unique user ID using uuid (this will be dynamic)
#     user_id = str(uuid.uuid4())  # Create a dynamic user ID

#     # Redirect to the WebSocket stream URL for that user
#     return RedirectResponse(url=f"/video_stream/{user_id}", status_code=303)  # 303 See Other to enforce GET method

# # WebSocket endpoint to stream video
# @app.websocket("/video_stream/{user_id}")
# async def video_stream(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket

#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     if not cap.isOpened():
#         await websocket.send_text("Failed to open the camera.")
#         return

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
# import cv2
# import asyncio
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io
# import uuid  # For generating dynamic user IDs

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize template rendering engine
# templates = Jinja2Templates(directory="templates")

# # Initialize MediaPipe for hand gesture recognition
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # This will keep track of users' WebSocket connections
# user_sockets = {}

# # Route for connecting a user and getting their name
# @app.get("/connect_user/", response_class=HTMLResponse)
# async def connect_user_form(request: Request):
#     # Render the form to get the user's name
#     return templates.TemplateResponse("connect_user.html", {"request": request})

# # Route to handle the form submission and redirect to the video feed page with dynamic user ID
# @app.post("/connect_user/")
# async def connect_user(user_name: str = Form(...)):
#     # Generate a unique user ID using uuid (this will be dynamic)
#     user_id = str(uuid.uuid4())  # Create a dynamic user ID
    
#     # Redirect to the WebSocket stream URL for that user
#     return RedirectResponse(url=f"/video_stream/{user_id}", status_code=303)  # 303 See Other to enforce GET method

# # WebSocket endpoint to stream video
# @app.websocket("/video_stream/{user_id}")
# async def video_stream(websocket: WebSocket, user_id: str):
#     await websocket.accept()
#     user_sockets[user_id] = websocket

#     cap = cv2.VideoCapture(0)  # Each user gets their own camera
#     if not cap.isOpened():
#         await websocket.send_text("Failed to open the camera.")
#         return

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Flip and process frame
#             frame = cv2.flip(frame, 1)
#             framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(framergb)

#             height, width, _ = frame.shape
#             className = ""

#             if result.multi_hand_landmarks:
#                 for handslms in result.multi_hand_landmarks:
#                     landmarks = []
#                     x_coords = []
#                     y_coords = []

#                     for lm in handslms.landmark:
#                         lmx = int(lm.x * width)
#                         lmy = int(lm.y * height)
#                         landmarks.append([lmx, lmy])
#                         x_coords.append(lmx)
#                         y_coords.append(lmy)

#                     # Calculate bounding box
#                     x_min, x_max = min(x_coords), max(x_coords)
#                     y_min, y_max = min(y_coords), max(y_coords)
#                     cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                     # Draw landmarks
#                     mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                     # Reshape landmarks to match model input (21, 2)
#                     landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                    
#                     # Predict gesture
#                     if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                         prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                         classID = np.argmax(prediction)
#                         className = classNames[classID]

#             # Show prediction on frame
#             cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Encode frame to JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             await websocket.send_bytes(frame)
            
#             await asyncio.sleep(0.03)  # Adjust to control frame rate (e.g., 30 FPS)
#     except WebSocketDisconnect:
#         del user_sockets[user_id]
#         cap.release()

# # Health check endpoint
# @app.get("/health/")
# async def health_check():
#     return {"status": "OK"}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Response
# from fastapi.responses import StreamingResponse
# import uvicorn
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize MediaPipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# # Load class names
# with open('gesture.names', 'r') as f:
#     classNames = f.read().splitlines()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Flip and process frame
#         frame = cv2.flip(frame, 1)
#         framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(framergb)

#         height, width, _ = frame.shape
#         className = ""

#         if result.multi_hand_landmarks:
#             for handslms in result.multi_hand_landmarks:
#                 landmarks = []
#                 x_coords = []
#                 y_coords = []

#                 for lm in handslms.landmark:
#                     lmx = int(lm.x * width)
#                     lmy = int(lm.y * height)
#                     landmarks.append([lmx, lmy])
#                     x_coords.append(lmx)
#                     y_coords.append(lmy)

#                 # Calculate bounding box
#                 x_min, x_max = min(x_coords), max(x_coords)
#                 y_min, y_max = min(y_coords), max(y_coords)
#                 cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

#                 # Draw landmarks
#                 mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#                 # Reshape landmarks to match model input (21, 2)
#                 landmarks = np.array(landmarks).reshape(-1, 2)  # Ensure shape is (21, 2)
                
#                 # Predict gesture
#                 if landmarks.shape == (21, 2):  # Ensure the shape is correct before passing to model
#                     prediction = model.predict(np.expand_dims(landmarks, axis=0))  # Add batch dimension
#                     classID = np.argmax(prediction)
#                     className = classNames[classID]

#         # Show prediction on frame
#         cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Encode frame to JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.get("/")
# def index():
#     return {"message": "Welcome to the Hand Gesture Recognition API"}

# @app.get("/video_feed")
# def video_feed():
#     return StreamingResponse(generate_frames(),
#                              media_type="multipart/x-mixed-replace; boundary=frame")

# # Close the webcam when server shuts down
# @app.on_event("shutdown")
# def shutdown_event():
#     cap.release()

# # Run the server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Load the gesture recognizer model
model = load_model(r'D:\Sign_language_mediapipe\mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().splitlines()

# Initialize MediaPipe
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

# HTML for testing
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Hand Gesture Recognition</title>
</head>
<body>
    <h1>Hand Gesture Recognition</h1>
    <video id="video" autoplay></video>
    <script>
        const video = document.getElementById('video');
        const ws = new WebSocket('ws://localhost:8000/ws');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        document.body.appendChild(canvas);

        ws.onmessage = (event) => {
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(url);
            };
            img.src = url;
        };

        ws.onopen = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            const track = stream.getVideoTracks()[0];
            const imageCapture = new ImageCapture(track);
            
            async function captureFrame() {
                const bitmap = await imageCapture.grabFrame();
                canvas.width = bitmap.width;
                canvas.height = bitmap.height;
                ctx.drawImage(bitmap, 0, 0);
                canvas.toBlob(blob => {
                    ws.send(blob);
                }, 'image/jpeg');
                requestAnimationFrame(captureFrame);
            }
            captureFrame();
        };
    </script>
</body>
</html>
"""

# Client Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_data(self, data, websocket: WebSocket):
        await websocket.send_bytes(data)

manager = ConnectionManager()

@app.get("/")
def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    cap = cv2.VideoCapture(0)  # Create a unique instance for each connection
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    try:
        while True:
            data = await websocket.receive_bytes()  # Receive frame from client
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                    landmarks = np.array(landmarks).reshape(-1, 2)
                    if landmarks.shape == (21, 2):
                        prediction = model.predict(np.expand_dims(landmarks, axis=0))
                        classID = np.argmax(prediction)
                        className = classNames[classID]

            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            _, buffer = cv2.imencode('.jpg', frame)
            await manager.send_data(buffer.tobytes(), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        cap.release()
