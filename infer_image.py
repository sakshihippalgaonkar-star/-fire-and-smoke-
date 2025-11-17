# infer_image.py
import os
import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "fire_smoke_model.h5"
MAPPING_PATH = "class_map.json"
IMG_SIZE = (128,128)

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run train_fire.py first to create fire_smoke_model.h5")
if not os.path.exists(MAPPING_PATH):
    raise SystemExit("Mapping file not found. Run train_fire.py to create class_map.json")

model = load_model(MODEL_PATH)
with open(MAPPING_PATH, 'r') as f:
    index_to_class = json.load(f)
index_to_class = {int(k): v for k,v in index_to_class.items()}

def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Cannot read image:", path); return
    input_img = cv2.resize(img, IMG_SIZE)
    input_img = input_img.astype('float32')/255.0
    input_batch = np.expand_dims(input_img, axis=0)
    pred = model.predict(input_batch)[0]
    cls = int(pred.argmax())
    prob = float(pred[cls])
    label = index_to_class.get(cls, str(cls))
    print(f"Prediction: {label} (score={prob:.3f})")
    # show result
    disp = img.copy()
    cv2.putText(disp, f"{label} {prob:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Result", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if _name_ == "_main_":
    if len(sys.argv) < 2:
        print("Usage: python infer_image.py path/to/image.jpg")
    else:
        predict_image(sys.argv[1])