from fastapi import FastAPI, WebSocket  # Add WebSocket import here
from app.object_detection import websocket_endpoint

app = FastAPI()

# Define a route for basic testing
@app.get("/")
def read_root():
    return {"message": "FastAPI Object Detection API"}

# Add WebSocket endpoint for real-time detection
@app.websocket("/ws/detect/")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)  # Pass the WebSocket to the handler function
