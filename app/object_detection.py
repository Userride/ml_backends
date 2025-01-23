from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from app.yolo import YOLO_Pred
import base64
import numpy as np
from PIL import Image
import io

# Path to models
MODEL_PATH = 'app/ml_models/best.onnx'
DATA_PATH = 'app/ml_models/data.yaml'

# Initialize YOLO model
yolo = YOLO_Pred(MODEL_PATH, DATA_PATH)

# WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    
    try:
        while True:
            data = await websocket.receive_text()  # Receive the data sent by the client
            text_data_json = json.loads(data)
            image_data = text_data_json.get('image')

            if not image_data:
                continue

            image = decode_base64_image(image_data)

            # Run YOLO Prediction
            _, detect_res = yolo.predictions(image)

            # Compute Manhattan distances
            distances = compute_manhattan_distance(detect_res)

            response_data = {
                'detections': detect_res,
                'distances': distances,
            }

            await websocket.send_text(json.dumps(response_data))  # Send response back to the client
            await asyncio.sleep(0.02)  # Small delay for handling real-time frames

    except WebSocketDisconnect:
        print("Client disconnected")


def decode_base64_image(img_str: str):
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))
    return np.array(image)


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
