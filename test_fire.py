# test_fire.py — Debug version to check model and image paths

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys
import traceback

# STEP 1: Automatically detect the current project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# STEP 2: Model and image paths
MODEL_PATH = os.path.join(PROJECT_DIR, "fire_smoke_model.h5")
IMG_PATH = os.path.join(PROJECT_DIR, "dataset", "smoke", "smoke4.jpg")

# STEP 3: Print everything for debugging
print("PROJECT_DIR =", PROJECT_DIR)
print("MODEL_PATH  =", MODEL_PATH)
print("IMG_PATH    =", IMG_PATH)
print("\nFiles in project folder:")
for item in os.listdir(PROJECT_DIR):
    print("  ", item)

print("\nModel exists? ", os.path.exists(MODEL_PATH))
print("Image exists? ", os.path.exists(IMG_PATH))
print("-" * 40)

# STEP 4: Stop if files not found
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model file not found. Check the filename and folder.")
    sys.exit(1)

if not os.path.exists(IMG_PATH):
    print("❌ ERROR: Test image not found. Check dataset path and file name.")
    sys.exit(1)

# STEP 5: Load model
try:
    print("Loading model ...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model failed to load. Full traceback below:\n")
    traceback.print_exc()
    sys.exit(1)

# STEP 6: Preprocess image and make prediction
from tensorflow.keras.preprocessing import image
import numpy as np

print("Processing image for prediction...")

# Load and resize the image to match your model input size
img = image.load_img(IMG_PATH, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

# Make prediction
prediction = model.predict(img_array)
predicted_class = "Fire" if prediction[0][0] > 0.5 else "Smoke"

print(f"🔥 Predicted class: {predicted_class}")
print(f"Prediction value: {prediction[0][0]}")