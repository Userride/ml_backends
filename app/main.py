from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.object_detection import websocket_endpoint

app = FastAPI()

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

# Define a route for basic testing
@app.get("/")
def read_root():
    return {"message": "FastAPI Object Detection API"}

# Health check endpoint
@app.get("/notify/v1/health")
def get_health():
    return {"msg": "OK"}

# Add WebSocket endpoint for real-time detection
@app.websocket("/ws/detect/")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)  # Pass the WebSocket to the handler function
