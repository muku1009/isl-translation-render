import os
import cv2
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model

import gdown

# ================= CONFIG =================
BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "isl_cnn.h5")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 128
MAX_FRAMES = 15

# ================= DOWNLOAD MODEL =================
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    url = "https://drive.google.com/uc?id=1SW3n0yB7zm8qM24aJyIlYUbblb6gzA8t"
    gdown.download(url, MODEL_PATH, quiet=False)

# ================= LOAD MODEL =================
if not os.path.exists(MODEL_PATH):
    raise Exception("❌ Model not found!")

if not os.path.exists(LABELS_FILE):
    raise Exception("❌ labels.json not found!")

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_FILE, "r") as f:
    labels_list = json.load(f)

LABELS = {i: lbl for i, lbl in enumerate(labels_list)}

print("✅ Model Loaded Successfully")

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

# ================= HELPERS =================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(img):
    img = detect_and_crop_hand(img)
    img = preprocess(img)
    preds = model(img, training=False).numpy()[0]   # 🔥 replace predict()
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
    try:
        print("📩 Request received")

        file = request.files.get("file")

        if file is None:
            return jsonify({"error": "No file uploaded"})

        file_bytes = file.read()
        print("📦 File size:", len(file_bytes))

        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Image decoding failed"})

        print("🖼 Image shape:", img.shape)

        preds = predict_frame(img)

        idx = int(np.argmax(preds))
        print("✅ Prediction done")

        return jsonify({"prediction": LABELS[idx]})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)})

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)