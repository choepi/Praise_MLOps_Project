import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --------------------------------------
# Feature Extraction Helpers
# --------------------------------------

def distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))

def extract_hand_features(landmarks):
    MCP = [5, 9, 13, 17]
    palm_center = type('obj', (), {
        'x': np.mean([landmarks[i].x for i in MCP]),
        'y': np.mean([landmarks[i].y for i in MCP]),
        'z': np.mean([landmarks[i].z for i in MCP])
    })
    palm_width = distance(landmarks[5], landmarks[17])
    if palm_width == 0:
        palm_width = 1e-6  # Avoid divide-by-zero

    return np.array([
        distance(landmarks[4], palm_center) / palm_width,
        distance(landmarks[8], palm_center) / palm_width,
        distance(landmarks[12], palm_center) / palm_width,
        distance(landmarks[16], palm_center) / palm_width,
        distance(landmarks[20], palm_center) / palm_width,
        calculate_angle(landmarks[2], landmarks[3], landmarks[4]),
        calculate_angle(landmarks[6], landmarks[7], landmarks[8]),
        calculate_angle(landmarks[10], landmarks[11], landmarks[12]),
        calculate_angle(landmarks[14], landmarks[15], landmarks[16]),
        calculate_angle(landmarks[18], landmarks[19], landmarks[20]),
    ], dtype=np.float32)

# --------------------------------------
# Dataset
# --------------------------------------

class HandDataset(Dataset):
    def __init__(self, dataset_split, features):
        self.dataset = dataset_split
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), self.dataset[idx]['label']

def precompute_features(dataset_split):
    features = []
    for sample in tqdm(dataset_split, desc="Extracting features"):
        img = np.array(sample['image'].convert('RGB'))
        results = hands.process(img)
        if not results.multi_hand_landmarks:
            features.append(np.zeros(10, dtype=np.float32))
        else:
            features.append(extract_hand_features(results.multi_hand_landmarks[0].landmark))
    return np.array(features)

# --------------------------------------
# Model
# --------------------------------------

class GestureClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------
# Train & Evaluate
# --------------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")
    return model

# --------------------------------------
# Export to ONNX
# --------------------------------------

def export_to_onnx(model, input_size=10):
    model.eval()
    dummy_input = torch.randn(1, input_size).to(device)
    export_path = os.path.join("..", "deployment", "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print(f"✅ Exported ONNX model to: {export_path}")

# --------------------------------------
# Main pipeline
# --------------------------------------

def main():
    dataset = load_dataset("Javtor/rock-paper-scissors")
    split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_split, val_split = split['train'], split['test']
    train_feats = precompute_features(train_split)
    val_feats = precompute_features(val_split)

    train_loader = DataLoader(HandDataset(train_split, train_feats), batch_size=32, shuffle=True)
    val_loader = DataLoader(HandDataset(val_split, val_feats), batch_size=32)

    model = GestureClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    export_to_onnx(model)

    hands.close()
    print("✅ Training pipeline complete.")

if __name__ == "__main__":
    main()
