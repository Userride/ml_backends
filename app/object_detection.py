from fastapi import FastAPI, WebSocket, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from io import BytesIO
from yolo_predictions import YOLO_Pred

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_config = YOLO_Pred(model_path="models/best.pt", data_path="models/data.yaml")

@app.get("/")
def read_root():
    return {"message": "FastAPI Object Detection API"}

@app.get("/notify/v1/health")
def get_health():
    return {"msg": "OK"}

@app.post("/api/image-detection")
async def process_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        detections, annotated_img = model_config.predict_image(image_data)

        img_bytes = BytesIO()
        annotated_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        return {
            "detections": detections,
            "image_bytes": img_bytes.read().hex(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video-detection")
async def process_video(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        output_file_path = "processed_video.mp4"

        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        model_config.predict_video(temp_file_path, output_file_path)

        os.remove(temp_file_path)

        return {"message": "Video processed successfully.", "file_path": output_file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/live-detection")
async def websocket_live_detection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            img_bytes = await websocket.receive_bytes()
            detections, annotated_img = model_config.predict_image(img_bytes)

            img_buffer = BytesIO()
            annotated_img.save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            await websocket.send_json({"detections": detections})
            await websocket.send_bytes(img_buffer.read())
    except Exception as e:
        await websocket.close(reason=str(e))

def clean_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
