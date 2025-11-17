# train_fire.py
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# CONFIG
DATA_DIR = r"C:\Users\saksh\OneDrive\Desktop\fire and smoke\dataset" # must contain subfolders e.g. data/fire, data/smoke
import os
print("Path exists:", os.path.exists(DATA_DIR))
print("Files inside dataset:", os.listdir(DATA_DIR))
MODEL_OUT = "fire_smoke_model.h5"
MAPPING_OUT = "class_map.json"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# Data generator (with simple augmentation) and validation split
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save mapping index->class name for inference
class_indices = train_gen.class_indices
print("Class indices (folder -> index):", class_indices)
index_to_class = {v: k for k, v in class_indices.items()}
with open(MAPPING_OUT, 'w') as f:
    json.dump(index_to_class, f)
print(f"Saved class mapping to {MAPPING_OUT}")

num_classes = train_gen.num_classes

# Build a small CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE + (3,)),
    BatchNormalization(),
    MaxPool2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save training plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.legend(); plt.title('Accuracy')

plt.tight_layout()
plt.savefig("training_plot.png")
print("Training complete. Model saved to", MODEL_OUT)