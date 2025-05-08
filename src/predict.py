# preprocess.py
import mediapipe as mp
import cv2
import numpy as np
from datasets import load_dataset

# 1. Load the Rock-Paper-Scissors image dataset from Hugging Face
dataset = load_dataset("Javtor/rock-paper-scissors")
train_ds = dataset["train"]
test_ds  = dataset["test"]

# 2. Initialize MediaPipe Hands for static image mode
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, 
                       max_num_hands=1, 
                       model_complexity=1,
                       min_detection_confidence=0.5)

def extract_landmarks(image):
    """Run MediaPipe on an image and return normalized (x,y) landmarks."""
    # Convert image to BGR (MediaPipe uses OpenCV’s default color space)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = hands.process(img)
    if not results.multi_hand_landmarks:
        return None  # no hand detected
    # Assume the first detected hand is the one we want (there should be one hand per image in this dataset)
    hand_landmarks = results.multi_hand_landmarks[0]
    # Extract (x, y, z) coordinates of each landmark as numpy array
    coords = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
    # Normalize landmarks: translate so wrist (index 0) is origin
    wrist = coords[0].copy()
    coords -= wrist  # subtract wrist coords from all points (broadcasting per row)
    # Drop the z dimension:
    coords = coords[:, :2]  # use only x and y for features
    # Scale landmarks: find max distance from origin (wrist) and scale to unit length
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords

# 3. Loop through dataset and build feature matrices and labels
X_train, y_train = [], []
for sample in train_ds:
    img = sample["image"]
    label = sample["label"]            # 0,1,2 for paper, rock, scissors
    feats = extract_landmarks(img)
    if feats is not None:
        # Flatten to a 1D feature vector with 42 values
        X_train.append(feats.flatten())
        y_train.append(label)

X_test, y_test = [], []
for sample in test_ds:
    img = sample["image"]
    label = sample["label"]
    feats = extract_landmarks(img)
    if feats is not None:
        # Flatten to a 1D feature vector with 42 values
        X_test.append(feats.flatten())
        y_test.append(label)

X_train = np.array(X_train, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)


# 4. Save the processed data to disk for use in training
np.savez("landmarks_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Saved normalized landmark features to landmarks_data.npz")
