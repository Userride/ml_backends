from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from yolo_predictions import YOLO_Pred
from starlette.responses import Response, StreamingResponse
import io
from PIL import Image
import numpy as np
import json
import cv2
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import base64

# Initialize and obtain the model
yolo = YOLO_Pred('app/ml_models/best.onnx', 'app/ml_models/data.yaml')

# FastAPI application setup
app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

# CORS (Cross-Origin Resource Sharing) middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')


# Convert bytes to a NumPy array
def bytes_to_numpy(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    return np.array(image)


def compute_manhattan_distance(detections):
    """Compute Manhattan distances between detected objects."""
    distances = []
    num_objects = len(detections)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1, obj2 = detections[i], detections[j]
            x1_center = (obj1['x1'] + obj1['x2']) // 2
            y1_center = (obj1['y1'] + obj1['y2']) // 2
            x2_center = (obj2['x1'] + obj2['x2']) // 2
            y2_center = (obj2['y1'] + obj2['y2']) // 2

            distance = abs(x1_center - x2_center) + abs(y1_center - y2_center)
            distances.append({
                "object1": obj1['label'],
                "object2": obj2['label'],
                "distance": distance,
            })
    return distances


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    # Convert bytes to NumPy array
    input_image = bytes_to_numpy(file)
    results = yolo.predictions(input_image)

    # Convert results to a list of detected objects
    detect_res = results["detections"]  # Assuming your YOLO model returns detections in this format

    # Compute Manhattan distances between detected objects
    distances = compute_manhattan_distance(detect_res)

    return {
        "detections": detect_res,
        "distances": distances
    }


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    # Convert bytes to NumPy array
    input_image = bytes_to_numpy(file)
    # Get processed image and detections
    processed_image, detect_res = yolo.predictions(input_image)

    # Compute Manhattan distances between detected objects
    distances = compute_manhattan_distance(detect_res)

    # Convert processed image to base64
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(processed_image)
    img_base64.save(bytes_io, format="jpeg")
    img_str = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
    
    return {
        "image": img_str,
        "detections": detect_res,
        "distances": distances
    }


@app.post("/object-to-video")
async def detect_objects_from_video(file: UploadFile = File(...)):
    # Save the uploaded video file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with temp_file as f:
            f.write(await file.read())

        # Open the video file with OpenCV
        video = cv2.VideoCapture(temp_file.name)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # Prepare output video stream
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # YOLO prediction on each frame
            processed_frame, detect_res = yolo.predictions(frame)

            # Save the result frame
            out.write(processed_frame)

        video.release()
        out.release()

        # Return the processed video
        return StreamingResponse(io.BytesIO(open('output.mp4', 'rb').read()), media_type="video/mp4")

    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)
        if os.path.exists('output.mp4'):
            os.remove('output.mp4')


@app.websocket("/ws/realtime-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive frame data from the client
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data))
            frame = np.array(image)

            # Process the frame with YOLO
            processed_frame, detect_res = yolo.predictions(frame)

            # Compute Manhattan distances
            distances = compute_manhattan_distance(detect_res)

            # Convert results to an image
            result_image = Image.fromarray(processed_frame)
            buffer = io.BytesIO()
            result_image.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()

            detection_info = {
                "detections": detect_res,
                "distances": distances
            }

            # Send processed frame back to the client
            await websocket.send_bytes(img_bytes)
            await websocket.send_text(json.dumps(detection_info))

        except WebSocketDisconnect:
            print("Client disconnected")
            break
