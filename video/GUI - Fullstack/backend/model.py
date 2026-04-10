"""
Cricket Shot Classifier – Model & Preprocessing
Extracted from the training notebook for standalone inference.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms


# ── Class labels (sorted, matching training) ────────────────────────────────
CLASS_NAMES = ["fast_left", "fast_right", "leg_left", "leg_right", "off_left", "off_right"]


# ── Model architecture (identical to training notebook) ─────────────────────
class CricketShotBaseline(nn.Module):
    def __init__(self, num_classes=6, pretrained=False):
        super(CricketShotBaseline, self).__init__()
        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        features = self.resnet(x)
        features = features.view(batch_size, sequence_length, -1)
        pooled_features = torch.mean(features, dim=1)
        output = self.fc(pooled_features)
        return output


# ── Preprocessing (mirrors BowlingVideoDataset.extract_frames) ──────────────
def extract_frames(video_path: str, sequence_length: int = 16) -> list:
    """Extract *sequence_length* uniformly-spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read any frames from {video_path}")

    if total_frames >= sequence_length:
        indices = set(np.linspace(0, total_frames - 1, sequence_length, dtype=int))
    else:
        indices = set(np.concatenate((
            np.arange(total_frames),
            np.full(max(0, sequence_length - total_frames), total_frames - 1),
        )))

    frames: list[Image.Image] = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()

    # Pad if needed
    while len(frames) < sequence_length:
        frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))

    return frames[:sequence_length]


def get_transforms() -> transforms.Compose:
    """Return the same transform pipeline used during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess_video(video_path: str) -> torch.Tensor:
    """Full pipeline: video file → model-ready tensor of shape (1, 16, 3, 224, 224)."""
    frames = extract_frames(video_path)
    transform = get_transforms()
    tensor_frames = torch.stack([transform(f) for f in frames])  # (16, 3, 224, 224)
    return tensor_frames.unsqueeze(0)  # (1, 16, 3, 224, 224)
