from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
import base64
import numpy as np
from PIL import Image
import io
import os
import tempfile
import cv2
from app.yolo import YOLO_Pred

# Path to models
MODEL_PATH = 'app/ml_models/best.onnx'
DATA_PATH = 'app/ml_models/data.yaml'

# Initialize YOLO model
yolo = YOLO_Pred(MODEL_PATH, DATA_PATH)

app = FastAPI()

# CORS Middleware
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
    return {"msg": "OK"}

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            text_data_json = json.loads(data)
            image_data = text_data_json.get('image')

            if not image_data:
                continue

            image = decode_base64_image(image_data)
            _, detect_res = yolo.predictions(image)
            distances = compute_manhattan_distance(detect_res)

            response_data = {
                'detections': detect_res,
                'distances': distances,
            }

            await websocket.send_text(json.dumps(response_data))
            await asyncio.sleep(0.02)  # Small delay for handling real-time frames

    except WebSocketDisconnect:
        print("Client disconnected")

# Utility Functions
def bytes_to_numpy(data: bytes) -> np.ndarray:
    """Convert bytes to a NumPy array."""
    image = Image.open(io.BytesIO(data))
    return np.array(image)

def decode_base64_image(img_str: str):
    """Decode base64 image string into a NumPy array."""
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))
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
async def detect_objects_return_json(file: bytes = File(...)):
    """Detect objects and return results in JSON format."""
    input_image = bytes_to_numpy(file)
    results = yolo.predictions(input_image)
    detect_res = results.tolist()
    return {"result": detect_res}

@app.post("/object-to-img")
async def detect_objects_return_base64_img(file: bytes = File(...)):
    """Detect objects and return processed image as base64."""
    input_image = bytes_to_numpy(file)
    processed_image, detect_res = yolo.predictions(input_image)

    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(processed_image)
    img_base64.save(bytes_io, format="jpeg")
    img_str = base64.b64encode(bytes_io.getvalue()).decode('utf-8')

    return {"image": img_str, "detections": detect_res}

@app.post("/object-to-video")
async def detect_objects_from_video(file: UploadFile = File(...)):
    """Detect objects in a video file and return processed video."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        with temp_file as f:
            f.write(await file.read())

        video = cv2.VideoCapture(temp_file.name)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = "output.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            results = yolo.predictions(frame)
            out.write(results)

        video.release()
        out.release()

        return StreamingResponse(io.BytesIO(open(output_path, 'rb').read()), media_type="video/mp4")

    finally:
        os.remove(temp_file.name)
        if os.path.exists(output_path):
            os.remove(output_path)
