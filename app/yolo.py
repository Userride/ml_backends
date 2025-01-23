# app/yolo.py
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def _init_(self, onnx_model, data_yaml):
        # Load YAML file
        try:
            with open(data_yaml, mode='r') as f:
                data_yaml = yaml.load(f, Loader=SafeLoader)
        except Exception as e:
            print(f"Error loading YAML file: {e}")
            return

        self.labels = data_yaml.get('names', [])
        self.nc = data_yaml.get('nc', 0)

        # Load YOLO model
        try:
            self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return
        
    def predictions(self, image):
        try:
            row, col, d = image.shape
        except AttributeError:
            print("Invalid image format")
            return image, []

        # Convert image into square
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        
        # Get prediction from square image
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Non-Maximum Suppression
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        detected_objects = []

        # NMS
        if len(boxes_np) > 0:
            index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
            if index is None or len(index) == 0:
                return image, []

            index = index.flatten() if isinstance(index, np.ndarray) else index

            for ind in index:
                x, y, w, h = boxes_np[ind]
                bb_conf = int(confidences_np[ind] * 100)
                classes_id = classes[ind]
                class_name = self.labels[classes_id]
                colors = self.generate_colors(classes_id)

                position = "center"
                distance = "far"

                if x < image_w / 3:
                    position = "left"
                elif x > 2 * image_w / 3:
                    position = "right"

                if h > image_h / 2:
                    distance = "near"

                detected_objects.append({
                    "label": class_name,
                    "confidence": bb_conf,
                    "position": position,
                    "distance": distance,
                    "x1": x,  # Left coordinate
                    "y1": y,  # Top coordinate
                    "x2": x + w,  # Right coordinate
                    "y2": y + h  # Bottom coordinate
                })

                text = f'{class_name}: {bb_conf}%'

                cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
                cv2.rectangle(image, (x, y - 30), (x + w, y), colors, -1)

                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

        return image, detected_objects

    def generate_colors(self, ID):
        # Fixed color palette for each class
        color_palette = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (0, 255, 255),  # Yellow
            (255, 255, 0),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (0, 255, 127)   # SpringGreen
        ]
        return color_palette[ID % len(color_palette)]
