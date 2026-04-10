"""
Cricket Shot Classifier – FastAPI Inference Server
Enhanced with frame extraction, trajectory simulation, and processing metrics.
"""

import os
import time
import base64
import tempfile
import random
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import CricketShotBaseline, preprocess_video, extract_frames as extract_raw_frames, CLASS_NAMES

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Cricket Shot Classifier API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ──────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "cricket_bowling_model_ep15.pth")
device = torch.device("cpu")

model = CricketShotBaseline(num_classes=len(CLASS_NAMES), pretrained=False)
state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Model loaded from {WEIGHTS_PATH} on {device}")

# ── Training history (from notebook) ────────────────────────────────────────
TRAINING_HISTORY = {
    "epochs": list(range(1, 16)),
    "train_loss": [1.5848, 1.2923, 1.1000, 0.9088, 0.7800, 0.9433, 0.8823, 0.8260, 0.7521, 0.7099, 0.6443, 0.5793, 0.5406, 0.5132, 0.5011],
    "val_loss":   [2.2639, 1.0386, 1.2213, 0.9301, 0.7985, 0.8997, 0.8940, 0.8407, 0.8899, 0.7877, 0.7539, 0.7882, 0.7170, 0.7243, 0.7011],
    "train_acc":  [36.49, 53.55, 65.11, 78.33, 84.11, 76.87, 79.98, 82.60, 86.39, 87.85, 91.74, 94.70, 96.31, 97.47, 97.86],
    "val_acc":    [40.78, 71.65, 70.29, 80.58, 86.21, 77.48, 81.36, 85.05, 83.69, 86.60, 88.35, 88.35, 90.10, 89.51, 90.10],
}

# ── Trajectory templates per bowling action ─────────────────────────────────
def generate_trajectory(predicted_class, num_points=20):
    """Generate a simulated 2D ball trajectory based on the predicted bowling type."""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)  # 0..1

        # Base trajectory: top to bottom with slight curve
        y = t * 100  # 0 to 100 (pitch length)

        if "fast" in predicted_class:
            # Fast bowling: straighter, slight lateral drift
            swing = 15 * math.sin(t * math.pi * 0.7)
            x = 50 + (swing if "right" in predicted_class else -swing)
            speed = 140 + random.uniform(-5, 5)  # km/h
        elif "leg" in predicted_class:
            # Leg spin: curve after bounce
            if t < 0.6:
                x = 50 + (5 * t if "right" in predicted_class else -5 * t)
            else:
                curve = 25 * ((t - 0.6) ** 1.5)
                x = 50 + (curve + 3 if "right" in predicted_class else -(curve + 3))
            speed = 85 + random.uniform(-3, 3)
        else:  # off spin
            if t < 0.6:
                x = 50 + (-5 * t if "right" in predicted_class else 5 * t)
            else:
                curve = 20 * ((t - 0.6) ** 1.5)
                x = 50 + (-curve - 3 if "right" in predicted_class else curve + 3)
            speed = 80 + random.uniform(-3, 3)

        x += random.uniform(-1.5, 1.5)
        points.append({"x": round(x, 2), "y": round(y, 2), "speed": round(speed, 1)})

    return points


def get_keyframe_thumbnails(video_path, num_frames=8):
    """Extract keyframe thumbnails as base64 JPEG strings for the frontend grid."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        cap.release()
        return [], {"total_frames": 0, "fps": 0, "width": 0, "height": 0}

    indices = set(np.linspace(0, total - 1, num_frames, dtype=int))
    thumbnails = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            # Resize to thumbnail
            thumb = cv2.resize(frame, (224, 224))
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode('utf-8')
            thumbnails.append({"frame_index": int(i), "data": b64})

    cap.release()
    return thumbnails, {
        "total_frames": total,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration": round(total / fps, 2) if fps > 0 else 0,
    }


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Cricket Shot Classifier API v2.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload .mp4, .avi, .mov, or .mkv",
        )

    try:
        suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        start_time = time.time()

        # Extract keyframes for frontend display
        thumbnails, video_meta = get_keyframe_thumbnails(tmp_path, num_frames=8)

        # Preprocess and infer
        tensor = preprocess_video(tmp_path)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        inference_time = round(time.time() - start_time, 3)

        predicted_idx = int(torch.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])
        all_scores = {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)}

        # Generate trajectory
        trajectory = generate_trajectory(predicted_class)

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "trajectory": trajectory,
            "keyframes": thumbnails,
            "video_meta": video_meta,
            "inference_time_seconds": inference_time,
            "model_info": {
                "architecture": "ResNet-18",
                "input_shape": "16 × 3 × 224 × 224",
                "num_classes": 6,
                "class_names": CLASS_NAMES,
                "dataset_size": 2573,
                "val_accuracy": 90.10,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingWarmRestarts",
                "epochs_trained": 15,
            },
            "training_history": TRAINING_HISTORY,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
