# infer_webcam.py
import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "fire_smoke_model.h5"
MAPPING_PATH = "class_map.json"
IMG_SIZE = (128,128)

if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
    raise SystemExit("Run train_fire.py first to produce model and mapping.")

model = load_model(MODEL_PATH)
with open(MAPPING_PATH, 'r') as f:
    index_to_class = json.load(f)
index_to_class = {int(k): v for k,v in index_to_class.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Try closing other apps or use index 1.")

print("Press ESC to exit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, IMG_SIZE)
    inp = img.astype('float32')/255.0
    inp = np.expand_dims(inp, 0)
    pred = model.predict(inp)[0]
    cls = int(pred.argmax()); prob = float(pred[cls])
    label = index_to_class.get(cls, str(cls))
    label_low = label.lower()
    color = (0,0,255) if "fire" in label_low else ((0,255,255) if "smoke" in label_low else (0,255,0))
    cv2.putText(frame, f"{label} {prob:.2f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Webcam Fire & Smoke Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()