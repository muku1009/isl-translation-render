import os
import cv2
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================= CONFIG =================
BASE_DIR = r"F:\1. BE CSE AIML\SEM 8\CAPSTONE PROJECT SEM 8\CNN"

TRAIN_ONCE = False   # 🔴 True ONLY first training

IMAGE_TRAIN_DIR = os.path.join(BASE_DIR, "words_2")
IMAGE_VAL_DIR   = os.path.join(BASE_DIR, "words")

MODEL_PATH   = os.path.join(BASE_DIR, "isl_cnn.h5")
LABELS_FILE  = os.path.join(BASE_DIR, "labels.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 224
EPOCHS = 6
MAX_FRAMES = 15

# ================= FLASK =================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= HAND DETECTION =================
def detect_and_crop_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if w < 60 or h < 60:
        return frame

    return frame[y:y+h, x:x+w]

# ================= MODEL =================
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ================= TRAINING =================
if TRAIN_ONCE:
    print("🚀 Training CNN Model")

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        IMAGE_TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical"
    )

    val_gen = datagen.flow_from_directory(
        IMAGE_VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical"
    )

    labels = sorted(train_gen.class_indices.keys(), key=str)
    json.dump(labels, open(LABELS_FILE, "w"))

    model = build_model(len(labels))

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    model.save(MODEL_PATH)
    json.dump(history.history, open(HISTORY_FILE, "w"))
    print("✅ Training Complete")

# ================= LOAD MODEL =================
def load_model_robust(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        err = str(e)
        # If the error refers to InputLayer/batch_shape mismatch, try standalone keras
        if "batch_shape" in err or "InputLayer" in err:
            try:
                import keras as skkeras
                return skkeras.models.load_model(path)
            except Exception:
                pass
        # As a last resort define a compatible InputLayer that accepts 'batch_shape'
        class CompatibleInputLayer(tf.keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                if 'batch_shape' in config:
                    config['batch_input_shape'] = tuple(config.pop('batch_shape'))
                return super(CompatibleInputLayer, cls).from_config(config)

        # map dtype policy from standalone keras to tf.keras mixed precision Policy
        custom_map = {
            "InputLayer": CompatibleInputLayer,
            "DTypePolicy": tf.keras.mixed_precision.Policy
        }
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom_map)

try:
    model = load_model_robust(MODEL_PATH)
except Exception as e:
    print("Failed to load model:\n", e)
    raise

LABELS = json.load(open(LABELS_FILE))
LABELS = {i: lbl for i, lbl in enumerate(LABELS)}

# ================= HELPERS =================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(img):
    img = detect_and_crop_hand(img)
    preds = model.predict(preprocess(img), verbose=0)[0]
    return preds

def extract_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files.get("file")
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    preds = predict_frame(img)
    idx = int(np.argmax(preds))
    return jsonify({"prediction": LABELS[idx]})

@app.route("/predict_video", methods=["POST"])
def predict_video():
    file = request.files.get("file")
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    frames = extract_video_frames(path)
    avg_preds = np.zeros(len(LABELS))

    for f in frames:
        avg_preds += predict_frame(f)

    avg_preds /= len(frames)
    idx = int(np.argmax(avg_preds))
    os.remove(path)

    return jsonify({"prediction": LABELS[idx]})

# ================= MAIN =================
if __name__ == "__main__":
    print("APP_2 RUNNING 🚀")
    app.run(debug=True, use_reloader=False, port=5001)
