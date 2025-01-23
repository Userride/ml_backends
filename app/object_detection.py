from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from app.yolo import YOLO_Pred
import base64
import numpy as np
from PIL import Image
import io

# Paths to YOLO model files
MODEL_PATH = 'app/ml_models/best.onnx'
DATA_PATH = 'app/ml_models/data.yaml'

# Initialize YOLO model
yolo = YOLO_Pred(MODEL_PATH, DATA_PATH)

# WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    try:
        while True:
            # Wait for a message from the client
            data = await websocket.receive_text()
            
            # Parse the incoming JSON data
            try:
                text_data_json = json.loads(data)
                image_data = text_data_json.get('image')
                
                if not image_data:
                    await websocket.send_text(json.dumps({"error": "No image data provided"}))
                    continue

                # Decode base64 image to NumPy array
                image = decode_base64_image(image_data)
                if image is None:
                    await websocket.send_text(json.dumps({"error": "Invalid image data"}))
                    continue

                # Run YOLO prediction
                _, detect_res = yolo.predictions(image)

                # Compute Manhattan distances between detected objects
                distances = compute_manhattan_distance(detect_res)

                # Prepare and send response
                response_data = {
                    'detections': detect_res,
                    'distances': distances,
                }
                await websocket.send_text(json.dumps(response_data))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Processing error: {str(e)}"}))

            # Add a slight delay for real-time processing
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        await websocket.close()  # Ensure the WebSocket connection is closed


def decode_base64_image(img_str: str):
    """Decode a base64-encoded image string into a NumPy array."""
    try:
        img_data = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_data))
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def compute_manhattan_distance(detections):
    """Compute Manhattan distances between detected objects."""
    distances = []
    num_objects = len(detections)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1, obj2 = detections[i], detections[j]
            try:
                # Calculate centers of detected objects
                x1_center = (obj1['x1'] + obj1['x2']) // 2
                y1_center = (obj1['y1'] + obj1['y2']) // 2
                x2_center = (obj2['x1'] + obj2['x2']) // 2
                y2_center = (obj2['y1'] + obj2['y2']) // 2

                # Compute Manhattan distance
                distance = abs(x1_center - x2_center) + abs(y1_center - y2_center)
                distances.append({
                    "object1": obj1['label'],
                    "object2": obj2['label'],
                    "distance": distance,
                })
            except KeyError as e:
                print(f"Error computing distance: {e}")
                continue

    return distances
