from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import os
import io
import cv2
import numpy as np
import json
from PIL import Image
from app.yolo import YOLO_Pred
import base64
import tempfile

# Initialize YOLO model with paths
yolo = YOLO_Pred('app/ml_models/best.onnx', 'app/ml_models/data.yaml')

# FastAPI app initialization
app = FastAPI()

# CORS setup for handling cross-origin requests
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

@app.get("/")
def read_root():
    return {"message": "Object Detection API with Manhattan Distance"}

# Helper function to convert image bytes to numpy array
def bytes_to_numpy(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    return np.array(image)

# Compute Manhattan distance between detected objects
def compute_manhattan_distance(detections):
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
async def detect_objects_json(file: bytes = File(...)):
    input_image = bytes_to_numpy(file)
    results = yolo.predictions(input_image)
    detections = results["detections"]

    distances = compute_manhattan_distance(detections)

    return {
        "detections": detections,
        "distances": distances
    }

@app.post("/object-to-img")
async def detect_objects_image(file: bytes = File(...)):
    input_image = bytes_to_numpy(file)
    processed_image, detections = yolo.predictions(input_image)

    distances = compute_manhattan_distance(detections)

    # Convert the processed image to base64 string
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(processed_image)
    img_base64.save(bytes_io, format="JPEG")
    img_str = base64.b64encode(bytes_io.getvalue()).decode("utf-8")

    return {
        "image": img_str,
        "detections": detections,
        "distances": distances
    }

@app.post("/object-to-video")
async def detect_objects_video(file: UploadFile = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with temp_file as f:
            f.write(await file.read())

        video = cv2.VideoCapture(temp_file.name)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            processed_frame, detections = yolo.predictions(frame)
            out.write(processed_frame)

        video.release()
        out.release()

        return StreamingResponse(open(output_path, "rb"), media_type="video/mp4")

    finally:
        os.remove(temp_file.name)
        if os.path.exists("output.mp4"):
            os.remove("output.mp4")

# WebSocket for real-time object detection
@app.websocket("/ws/detect")
async def websocket_realtime_detection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive image frame as bytes
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data))
            frame = np.array(image)

            # Process the frame with YOLO
            processed_frame, detections = yolo.predictions(frame)
            distances = compute_manhattan_distance(detections)

            # Convert the processed frame back to bytes
            result_image = Image.fromarray(processed_frame)
            buffer = io.BytesIO()
            result_image.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()

            # Send the processed image and detections to client
            detection_info = {
                "detections": detections,
                "distances": distances
            }

            # Sending image and JSON information to the WebSocket client
            await websocket.send_bytes(img_bytes)
            await websocket.send_text(json.dumps(detection_info))

    except Exception as e:
        await websocket.close(reason=str(e))
