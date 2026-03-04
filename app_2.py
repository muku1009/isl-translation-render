import os
import cv2
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================= CONFIG =================
# BASE_DIR = r"F:\1. BE CSE AIML\SEM 8\CAPSTONE PROJECT SEM 8\DL BASED ISL TRANSLATION SYSTEM"
BASE_DIR = os.getcwd()

TRAIN_ONCE = False   # 🔴 Make True ONLY first time training

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

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ================= TRAINING =================
if TRAIN_ONCE:

    print("🚀 Training CNN Model...")

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

    labels = list(train_gen.class_indices.keys())
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)

    model = build_model(len(labels))

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    model.save(MODEL_PATH)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history.history, f)

    print("✅ Training Complete")

# ================= LOAD MODEL SAFELY =================
if not os.path.exists(MODEL_PATH):
    raise Exception("❌ Model file not found! Set TRAIN_ONCE = True first.")

if not os.path.exists(LABELS_FILE):
    raise Exception("❌ labels.json not found! Train model first.")

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_FILE, "r") as f:
    labels_list = json.load(f)

LABELS = {i: lbl for i, lbl in enumerate(labels_list)}

print("✅ Model Loaded Successfully")

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

    if file is None:
        return jsonify({"error": "No file uploaded"})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    preds = predict_frame(img)

    idx = int(np.argmax(preds))
    return jsonify({"prediction": LABELS[idx]})

@app.route("/predict_video", methods=["POST"])
def predict_video():
    file = request.files.get("file")

    if file is None:
        return jsonify({"error": "No file uploaded"})

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    frames = extract_video_frames(path)

    if len(frames) == 0:
        os.remove(path)
        return jsonify({"error": "No frames extracted from video"})

    avg_preds = np.zeros(len(LABELS))

    for f in frames:
        avg_preds += predict_frame(f)

    avg_preds /= len(frames)

    idx = int(np.argmax(avg_preds))

    os.remove(path)

    return jsonify({"prediction": LABELS[idx]})

# ================= MAIN =================
if __name__ == "__main__":
    print("🚀 APP_2 RUNNING on port 5001")
    app.run(debug=True, use_reloader=False, port=5001)