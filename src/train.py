import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
import wandb
import argparse
import hashlib
import glob
from pathlib import Path

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
print(" MediaPipe Hands initialized")

# Helper function to calculate Euclidean distance between 3D points
def distance(p1, p2):
    return math.sqrt(
        (p2.x - p1.x) ** 2 + 
        (p2.y - p1.y) ** 2 + 
        (p2.z - p1.z) ** 2
    )

# Helper function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    # Vectors from p2 to p1 and p2 to p3
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    # Dot product
    dot_product = np.dot(v1, v2)
    
    # Magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if mag1 * mag2 == 0:
        return 0
    
    # Angle in radians, then convert to degrees
    cos_angle = dot_product / (mag1 * mag2)
    # Clamp the value to avoid numerical issues
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

# Function to extract angle-invariant features from MediaPipe hand landmarks
def extract_hand_features(landmarks):
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_MCP = 5  # Base of index finger
    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_MCP = 9  # Base of middle finger
    RING_FINGER_TIP = 16
    RING_FINGER_MCP = 13  # Base of ring finger
    PINKY_TIP = 20
    PINKY_MCP = 17  # Base of pinky finger
    
    # Calculate palm center (average of base points of all fingers)
    palm_center_x = (landmarks[INDEX_FINGER_MCP].x + landmarks[MIDDLE_FINGER_MCP].x + 
                    landmarks[RING_FINGER_MCP].x + landmarks[PINKY_MCP].x) / 4
    palm_center_y = (landmarks[INDEX_FINGER_MCP].y + landmarks[MIDDLE_FINGER_MCP].y + 
                    landmarks[RING_FINGER_MCP].y + landmarks[PINKY_MCP].y) / 4
    palm_center_z = (landmarks[INDEX_FINGER_MCP].z + landmarks[MIDDLE_FINGER_MCP].z + 
                    landmarks[RING_FINGER_MCP].z + landmarks[PINKY_MCP].z) / 4
    
    palm_center = type('obj', (object,), {
        'x': palm_center_x, 
        'y': palm_center_y, 
        'z': palm_center_z
    })
    
    # Calculate palm width (distance between index MCP and pinky MCP)
    palm_width = distance(landmarks[INDEX_FINGER_MCP], landmarks[PINKY_MCP])
    
    # Feature 1: Finger extension ratios (distance from tip to palm center divided by palm width)
    thumb_extension = distance(landmarks[THUMB_TIP], palm_center) / palm_width
    index_extension = distance(landmarks[INDEX_FINGER_TIP], palm_center) / palm_width
    middle_extension = distance(landmarks[MIDDLE_FINGER_TIP], palm_center) / palm_width
    ring_extension = distance(landmarks[RING_FINGER_TIP], palm_center) / palm_width
    pinky_extension = distance(landmarks[PINKY_TIP], palm_center) / palm_width
    
    # Feature 2: Finger bending angles
    thumb_angle = calculate_angle(landmarks[2], landmarks[3], landmarks[4])  # IP joint
    index_angle = calculate_angle(landmarks[6], landmarks[7], landmarks[8])  # PIP joint
    middle_angle = calculate_angle(landmarks[10], landmarks[11], landmarks[12])  # PIP joint
    ring_angle = calculate_angle(landmarks[14], landmarks[15], landmarks[16])  # PIP joint
    pinky_angle = calculate_angle(landmarks[18], landmarks[19], landmarks[20])  # PIP joint
    
    # Feature 3: Inter-fingertip distances normalized by palm width
    index_to_middle_dist = distance(landmarks[INDEX_FINGER_TIP], landmarks[MIDDLE_FINGER_TIP]) / palm_width
    middle_to_ring_dist = distance(landmarks[MIDDLE_FINGER_TIP], landmarks[RING_FINGER_TIP]) / palm_width
    ring_to_pinky_dist = distance(landmarks[RING_FINGER_TIP], landmarks[PINKY_TIP]) / palm_width
    thumb_to_index_dist = distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_TIP]) / palm_width
    
    # Feature 4: Thumb opposition (how opposed the thumb is to other fingers)
    thumb_to_middle_dist = distance(landmarks[THUMB_TIP], landmarks[MIDDLE_FINGER_TIP]) / palm_width
    thumb_to_ring_dist = distance(landmarks[THUMB_TIP], landmarks[RING_FINGER_TIP]) / palm_width
    thumb_to_pinky_dist = distance(landmarks[THUMB_TIP], landmarks[PINKY_TIP]) / palm_width
    
    # Return all features as a numpy array
    return np.array([
        # Finger extension ratios
        thumb_extension, index_extension, middle_extension, ring_extension, pinky_extension,
        
        # Finger bending angles
        thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle,
        
        # Inter-fingertip distances
        index_to_middle_dist, middle_to_ring_dist, ring_to_pinky_dist, thumb_to_index_dist,
        
        # Thumb opposition
        thumb_to_middle_dist, thumb_to_ring_dist, thumb_to_pinky_dist
    ], dtype=np.float32)

# Function to load custom dataset from directory structure
def load_custom_dataset(root_dir):
    """
    Load custom dataset from directory structure.
    Expected structure:
    root_dir/
        rock/
            image1.jpg
            image2.png
            ...
        paper/
            image1.jpg
            ...
        scissor/
            image1.jpg
            ...
    """
    # Label mapping to match HuggingFace dataset convention
    label_mapping = {'paper': 0, 'rock': 1, 'scissor': 2}
    
    custom_data = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"⚠️ Custom dataset directory {root_dir} does not exist")
        return []
    
    print(f"📂 Loading custom dataset from {root_dir}")
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    for class_name in ['rock', 'paper', 'scissor']:
        class_dir = root_path / class_name
        if not class_dir.exists():
            print(f"⚠️ Directory {class_dir} not found, skipping...")
            continue
        
        # Find all image files in the class directory
        image_files = []
        for ext in extensions:
            image_files.extend(class_dir.glob(ext))
            image_files.extend(class_dir.glob(ext.upper()))  # Also check uppercase
        
        print(f"📸 Found {len(image_files)} images in {class_name}/ directory")
        
        for image_path in image_files:
            try:
                # Load and verify the image
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Add to dataset
                    custom_data.append({
                        'image': img.copy(),
                        'label': label_mapping[class_name],
                        'source': 'custom',
                        'path': str(image_path)
                    })
            
            except Exception as e:
                print(f"⚠️ Error loading image {image_path}: {e}")
                continue
    
    print(f"✅ Loaded {len(custom_data)} custom images")
    return custom_data

# Combined dataset class that handles both HuggingFace and custom data
class CombinedHandGestureDataset(Dataset):
    def __init__(self, hf_dataset=None, custom_data=None, precomputed_features=None):
        self.data = []
        self.precomputed_features = precomputed_features
        
        # Mapping of integer labels to class names
        self.idx_to_class = {0: 'paper', 1: 'rock', 2: 'scissor'}
        
        # Add HuggingFace data
        if hf_dataset is not None:
            for i, item in enumerate(hf_dataset):
                self.data.append({
                    'image': item['image'],
                    'label': item['label'],
                    'source': 'huggingface',
                    'index': i
                })
        
        # Add custom data
        if custom_data is not None:
            hf_len = len(hf_dataset) if hf_dataset else 0
            for i, item in enumerate(custom_data):
                self.data.append({
                    'image': item['image'],
                    'label': item['label'],
                    'source': 'custom',
                    'index': hf_len + i,
                    'path': item.get('path', '')
                })
        
        print(f"📊 Combined dataset: {len(self.data)} total samples")
        if hf_dataset:
            hf_count = len(hf_dataset)
            print(f"    📥 HuggingFace: {hf_count} samples")
        if custom_data:
            custom_count = len(custom_data)
            print(f"    📁 Custom: {custom_count} samples")
        
        # Print label distribution
        label_counts = {0: 0, 1: 0, 2: 0}
        for item in self.data:
            label_counts[item['label']] += 1
        
        print(f"📈 Label distribution:")
        for label, count in label_counts.items():
            print(f"    {self.idx_to_class[label]}: {count} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.precomputed_features is not None:
            # Use precomputed features
            features = self.precomputed_features[idx]
            label = self.data[idx]['label']
            return torch.FloatTensor(features), label
        
        # Get image from dataset
        item = self.data[idx]
        pil_image = item['image']
        
        # Convert RGBA to RGB if needed
        if pil_image.mode == 'RGBA':
            # Create a white background image
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            # Paste the image on the background using the alpha channel
            background.paste(pil_image, mask=pil_image.split()[3])  # 3 is the alpha channel
            pil_image = background
        elif pil_image.mode != 'RGB':
            # Convert any other mode to RGB
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        img = np.array(pil_image)
        
        # Process with MediaPipe to get hand landmarks
        results = hands.process(img)
        
        if not results.multi_hand_landmarks:
            # No hand detected, return zeros for features
            features = np.zeros(17, dtype=np.float32)
        else:
            # Extract angle-invariant features
            landmarks = results.multi_hand_landmarks[0].landmark
            features = extract_hand_features(landmarks)
        
        label = item['label']
        return torch.FloatTensor(features), label

# Keep the original class for backward compatibility
class HuggingFaceHandGestureDataset(Dataset):
    def __init__(self, dataset_split, precomputed_features=None):
        self.dataset = dataset_split
        self.precomputed_features = precomputed_features
        
        # Mapping of integer labels to class names (for debugging/display purposes)
        self.idx_to_class = {0: 'paper', 1: 'rock', 2: 'scissor'}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.precomputed_features is not None:
            # Use precomputed features
            features = self.precomputed_features[idx]
            label = self.dataset[idx]['label']  # Already an integer
            return torch.FloatTensor(features), label
        
        # Get image from dataset
        pil_image = self.dataset[idx]['image']
        
        # Convert RGBA to RGB if needed
        if pil_image.mode == 'RGBA':
            # Create a white background image
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            # Paste the image on the background using the alpha channel
            background.paste(pil_image, mask=pil_image.split()[3])  # 3 is the alpha channel
            pil_image = background
        elif pil_image.mode != 'RGB':
            # Convert any other mode to RGB
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        img = np.array(pil_image)
        
        # Process with MediaPipe to get hand landmarks
        results = hands.process(img)
        
        if not results.multi_hand_landmarks:
            # No hand detected, return zeros for features
            features = np.zeros(17, dtype=np.float32)
        else:
            # Extract angle-invariant features
            landmarks = results.multi_hand_landmarks[0].landmark
            features = extract_hand_features(landmarks)
        
        label = self.dataset[idx]['label']  # Already an integer
        return torch.FloatTensor(features), label

# Simple neural network classifier with highly parameterizable architecture
class GestureClassifier(nn.Module):
    def __init__(self, input_size=17, hidden_sizes=[32], num_classes=3, activation='relu', 
                 dropout_rate=0.2, batch_norm=False, use_residual=False):
        super(GestureClassifier, self).__init__()
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # Default
        
        # Build layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Add linear layer
            print(f"Adding layer {i+1}: {prev_size} -> {hidden_size}")
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add batch normalization if specified
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Add activation
            layers.append(self.activation)
            
            # Add dropout if rate > 0
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Store architecture info for residual connections
        self.hidden_sizes = hidden_sizes
        self.use_residual = use_residual
        self.has_residual = len(hidden_sizes) >= 2 and use_residual
        
        # If using residual connections, need separate modules for proper skip connections
        if self.has_residual:
            self.layers = nn.ModuleList()
            in_size = input_size
            
            for size in hidden_sizes:
                block = []
                block.append(nn.Linear(in_size, size))
                if batch_norm:
                    block.append(nn.BatchNorm1d(size))
                block.append(self.activation)
                if dropout_rate > 0:
                    block.append(nn.Dropout(dropout_rate))
                self.layers.append(nn.Sequential(*block))
                in_size = size
            
            self.output = nn.Linear(hidden_sizes[-1], num_classes)
            
            self.output_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        if not self.has_residual:
            # Standard feedforward network
            return self.model(x)
        else:
            # Network with residual connections
            prev_x = x
            for i, layer in enumerate(self.layers):
                if i > 0 and self.hidden_sizes[i] == self.hidden_sizes[i-1]:
                    # Add residual connection if layer sizes match
                    x = layer(x) + prev_x
                else:
                    x = layer(x)
                prev_x = x
            
            return self.output_activation(self.output(x))

# Function to precompute features for combined dataset
def precompute_features_combined(combined_dataset, batch_size=32, feature_groups=None, split_name="train"):
    print(f"🟡 Precomputing features for {len(combined_dataset)} samples ({split_name})...", flush=True)
    features_list = []
    start_time = time.time()
    
    # Determine feature size once upfront
    feature_size = 17  # Default full size
    if feature_groups is not None:
        feature_size = sum([
            5 if 'extension' in feature_groups else 0,    # 0-4
            5 if 'angles' in feature_groups else 0,       # 5-9
            4 if 'distances' in feature_groups else 0,    # 10-13
            3 if 'opposition' in feature_groups else 0    # 14-16
        ])
    
    # Process all images in batches
    total_processed = 0
    for i in tqdm(range(0, len(combined_dataset), batch_size), desc=f"📊 Extracting {split_name} features"):
        batch_images = []
        batch_start = time.time()
        
        # Prepare batch of images
        for j in range(i, min(i + batch_size, len(combined_dataset))):
            item = combined_dataset.data[j]
            pil_image = item['image']
            
            # Convert image format if needed
            if pil_image.mode == 'RGBA':
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[3])
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            batch_images.append(np.array(pil_image))
        
        # Process batch with MediaPipe
        for img in batch_images:
            results = hands.process(img)
            
            if not results.multi_hand_landmarks:
                # No hand detected
                features = np.zeros(feature_size, dtype=np.float32)
            else:
                # Extract features
                all_features = extract_hand_features(results.multi_hand_landmarks[0].landmark)
                
                # Filter features if needed
                if feature_groups is not None:
                    selected_indices = []
                    if 'extension' in feature_groups: selected_indices.extend(range(0, 5))
                    if 'angles' in feature_groups: selected_indices.extend(range(5, 10))
                    if 'distances' in feature_groups: selected_indices.extend(range(10, 14))
                    if 'opposition' in feature_groups: selected_indices.extend(range(14, 17))
                    
                    features = all_features[selected_indices]
                else:
                    features = all_features
            
            features_list.append(features)
        
        # Report progress
        total_processed += len(batch_images)
        elapsed = time.time() - start_time
        batch_time = time.time() - batch_start
        if i % 5 == 0 or i + batch_size >= len(combined_dataset):
            print(f"   ⏱️ [{split_name}] {total_processed}/{len(combined_dataset)} processed — {elapsed:.1f}s elapsed ({batch_time:.2f}s/batch)", flush=True)
    
    total_time = time.time() - start_time
    print(f"✅ Done with {split_name}: {len(features_list)} features in {total_time:.1f}s ({total_time/len(features_list)*1000:.1f}ms/sample)\n", flush=True)
    return np.array(features_list)

# Function to precompute features for all images
def precompute_features(dataset_split, batch_size=32, feature_groups=None, split_name="train"):
    print(f" Precomputing features for {len(dataset_split)} samples ({split_name})...", flush=True)
    features_list = []
    start_time = time.time()
    
    # Determine feature size once upfront
    feature_size = 17  # Default full size
    if feature_groups is not None:
        feature_size = sum([
            5 if 'extension' in feature_groups else 0,    # 0-4
            5 if 'angles' in feature_groups else 0,       # 5-9
            4 if 'distances' in feature_groups else 0,    # 10-13
            3 if 'opposition' in feature_groups else 0    # 14-16
        ])
    
    # Process all images in batches
    total_processed = 0
    for i in tqdm(range(0, len(dataset_split), batch_size), desc=f"📊 Extracting {split_name} features"):
        batch_images = []
        batch_start = time.time()
        
        # Prepare batch of images
        for j in range(i, min(i + batch_size, len(dataset_split))):
            pil_image = dataset_split[j]['image']
            
            # Convert image format if needed
            if pil_image.mode == 'RGBA':
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[3])
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            batch_images.append(np.array(pil_image))
        
        # Process batch with MediaPipe
        for img in batch_images:
            results = hands.process(img)
            
            if not results.multi_hand_landmarks:
                # No hand detected
                features = np.zeros(feature_size, dtype=np.float32)
            else:
                # Extract features
                all_features = extract_hand_features(results.multi_hand_landmarks[0].landmark)
                
                # Filter features if needed
                if feature_groups is not None:
                    selected_indices = []
                    if 'extension' in feature_groups: selected_indices.extend(range(0, 5))
                    if 'angles' in feature_groups: selected_indices.extend(range(5, 10))
                    if 'distances' in feature_groups: selected_indices.extend(range(10, 14))
                    if 'opposition' in feature_groups: selected_indices.extend(range(14, 17))
                    
                    features = all_features[selected_indices]
                else:
                    features = all_features
            
            features_list.append(features)
        
        # Report progress
        total_processed += len(batch_images)
        elapsed = time.time() - start_time
        batch_time = time.time() - batch_start
        if i % 5 == 0 or i + batch_size >= len(dataset_split):
            print(f"    [{split_name}] {total_processed}/{len(dataset_split)} processed — {elapsed:.1f}s elapsed ({batch_time:.2f}s/batch)", flush=True)
    
    total_time = time.time() - start_time
    print(f" Done with {split_name}: {len(features_list)} features in {total_time:.1f}s ({total_time/len(features_list)*1000:.1f}ms/sample)\n", flush=True)
    return np.array(features_list)

def get_feature_cache_path(dataset_name, split_name, batch_size, feature_groups, custom_path=None):
    """Generate a unique file path for cached features based on configuration."""
    # Create a unique identifier based on feature groups
    if feature_groups is None:
        feature_str = "all"
    else:
        feature_str = "_".join(sorted(feature_groups))
    
    # Include custom dataset info in cache path
    custom_str = ""
    if custom_path:
        custom_hash = hashlib.md5(str(custom_path).encode()).hexdigest()[:6]
        custom_str = f"_custom_{custom_hash}"
    
    # Create a hash of the configuration to keep filenames manageable
    config_str = f"{dataset_name}_{split_name}_{batch_size}_{feature_str}{custom_str}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:10]
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.getcwd(), "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, f"features_{config_hash}.npy")

def get_or_compute_features(dataset_split, split_name, batch_size=32, feature_groups=None, dataset_name="rps", custom_path=None):
    """Get features from cache if available, otherwise compute and cache them."""
    # Generate cache path
    cache_path = get_feature_cache_path(dataset_name, split_name, batch_size, feature_groups, custom_path)
    
    # Check if features are already cached
    if os.path.exists(cache_path):
        print(f" Loading cached features for {split_name} from {cache_path}")
        start_time = time.time()
        features = np.load(cache_path)
        print(f" Loaded {len(features)} {split_name} features in {time.time() - start_time:.2f}s")
        return features
    
    # Compute features
    print(f"🔄 No cached features found for {split_name}. Computing from scratch...")
    
    # Choose the appropriate function based on dataset type
    # Check if it's our CombinedHandGestureDataset class specifically
    if isinstance(dataset_split, CombinedHandGestureDataset):
        features = precompute_features_combined(dataset_split, batch_size, feature_groups, split_name)
    else:  # HuggingFace dataset or other standard dataset
        features = precompute_features(dataset_split, batch_size, feature_groups, split_name)
    
    # Cache the features
    print(f" Caching features to {cache_path}")
    np.save(cache_path, features)
    
    return features

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, 
                use_wandb=False, scheduler=None):
    print(" Starting training loop...")
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f"🏋️ Epoch {epoch+1}/{num_epochs} [Train]"):
            features, labels = features.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"🔍 Epoch {epoch+1}/{num_epochs} [Val]"):
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # For ReduceLROnPlateau
            else:
                scheduler.step()  # For other schedulers
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        print(f" Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # Log metrics to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch_time": epoch_time
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gesture_classifier.pth')
            print(f" New best model saved with accuracy: {best_val_acc:.4f}")
            
            # Log best model checkpoint to W&B
            if use_wandb:
                wandb.run.summary["best_val_accuracy"] = best_val_acc
                wandb.save('best_gesture_classifier.pth')
    
    total_time = time.time() - start_time
    print(f" Training finished in {total_time:.1f}s ({total_time/num_epochs:.1f}s/epoch)")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Log the training curve to W&B
    if use_wandb:
        wandb.log({"training_curves": wandb.Image('training_curves.png')})
    
    return model

# Function to export model to ONNX
def export_to_onnx(model, input_size=17, use_wandb=False):
    print(" Exporting model to ONNX format...")
    model.eval()
    dummy_input = torch.randn(1, input_size, device=device)
    
    onnx_file_path = "gesture_classifier.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(" Model exported to ONNX format")
    print(" Label mapping in exported model: output[0]=paper, output[1]=rock, output[2]=scissor")
    
    # Log ONNX model to W&B
    if use_wandb:
        # Create metadata file first
        metadata = {
            "framework": "onnx",
            "input_size": input_size,
            "label_mapping": {
                "0": "paper",
                "1": "rock",
                "2": "scissor"
            }
        }
        
        # Write metadata to a file
        metadata_file = "metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            f.write(json.dumps(metadata, indent=2))
        
        # Now create the artifact and add both files before logging
        print("📤 Uploading model to Weights & Biases...")
        artifact = wandb.Artifact("gesture_classifier_onnx", type="model")
        artifact.add_file(onnx_file_path)
        artifact.add_file(metadata_file)
        
        # Log the artifact with all files already added
        wandb.log_artifact(artifact)
        print(f" ONNX model uploaded to Weights & Biases: {wandb.run.name}")

# Main function to run the training pipeline
def main():
    print(" ENTERED MAIN FUNCTION")
    
    def parse_none(value):
        return None if value.lower() == 'none' else value
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Rock-Paper-Scissors Hand Gesture Classifier')
    
    # W&B parameters
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--team', type=str, default='praise_mlops', help='W&B team name')
    parser.add_argument('--project', type=str, default='rock-paper-scissors', help='W&B project name')
    parser.add_argument('--name', type=str, default=None, help='W&B run name')
    parser.add_argument('--sweep', type=lambda x: x.lower() == 'true', default=False, 
                        help='Run as part of a WandB sweep')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='Javtor/rock-paper-scissors',
                       help='HuggingFace dataset name to use')
    parser.add_argument('--custom_dataset', type=str, default=None,
                       help='Path to custom dataset directory with rock/paper/scissor subdirectories')
    parser.add_argument('--hf_only', action='store_true', 
                       help='Use only HuggingFace dataset (ignore custom dataset)')
    parser.add_argument('--custom_only', action='store_true',
                       help='Use only custom dataset (ignore HuggingFace dataset)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], 
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for SGD)')
    parser.add_argument('--scheduler', type=parse_none, default=None, 
                    choices=[None, 'step', 'cosine', 'plateau'],
                    help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    
    # Model architecture parameters
    parser.add_argument('--hidden_sizes', type=str, default='32', 
                        help='Comma-separated list of hidden layer sizes (e.g., "64,32")')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
                        help='Activation function')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_norm', type=lambda x: x.lower() == 'true', default=False, 
                        help='Use batch normalization')
    parser.add_argument('--residual', type=lambda x: x.lower() == 'true', default=False, 
                        help='Use residual connections')
    
    # Feature selection parameters
    parser.add_argument('--feature_groups', type=str, default='all',
                        help='Comma-separated list of feature groups to use: extension,angles,distances,opposition')
    
    # Export parameters
    parser.add_argument('--export_dir', type=str, default='.', 
                        help='Directory to export model files')
    
    args = parser.parse_args()
    
    # Validate dataset options
    if args.hf_only and args.custom_only:
        raise ValueError("Cannot use both --hf_only and --custom_only flags")
    
    if args.custom_only and not args.custom_dataset:
        raise ValueError("Must provide --custom_dataset path when using --custom_only")
    
    # Create export directory if it doesn't exist
    print(f" Using export directory: {args.export_dir}")
    os.makedirs(args.export_dir, exist_ok=True)
    
    # Parse hidden_sizes
    if args.sweep:
        # In sweep mode, we need to get config from wandb
        print(" Initializing sweep run...")
        if args.team:
            wandb.init(project=args.project, entity=args.team)
        else:
            wandb.init(project=args.project)
            
        # Get hidden_sizes from wandb config
        hidden_sizes_raw = wandb.config.get('hidden_sizes', args.hidden_sizes)
        print(f" Hidden sizes from sweep config: {hidden_sizes_raw}")
    else:
        # Normal mode - get from args
        hidden_sizes_raw = args.hidden_sizes
    
    # Now properly parse hidden_sizes regardless of source
    if isinstance(hidden_sizes_raw, str):
        if ',' in hidden_sizes_raw:
            hidden_sizes = [int(size.strip()) for size in hidden_sizes_raw.split(',')]
        else:
            hidden_sizes = [int(hidden_sizes_raw.strip())]
    elif isinstance(hidden_sizes_raw, (list, tuple)):
        hidden_sizes = [int(size) for size in hidden_sizes_raw]
    elif isinstance(hidden_sizes_raw, (int, float)):
        hidden_sizes = [int(hidden_sizes_raw)]
    else:
        print(f" WARNING: Unexpected hidden_sizes format: {type(hidden_sizes_raw)}. Using default [32]")
        hidden_sizes = [32]
    
    print(f" Using hidden sizes: {hidden_sizes}")
    
    # Parse feature groups
    if args.feature_groups.lower() == 'all':
        feature_groups = None  # Use all features
        print(" Using all feature groups")
    else:
        feature_groups = args.feature_groups.split(',')
        print(f" Using selected feature groups: {feature_groups}")
    
    # Initialize Weights & Biases
    if args.wandb or args.sweep:
        print(" Setting up Weights & Biases...")
        # Create config dictionary
        wandb_config = {
            # Training parameters
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            
            # Optimizer parameters
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "scheduler": args.scheduler,
            "step_size": args.step_size,
            "gamma": args.gamma,
            
            # Model architecture parameters
            "hidden_sizes": hidden_sizes,
            "activation": args.activation,
            "dropout_rate": args.dropout_rate,
            "batch_norm": args.batch_norm,
            "residual": args.residual,
            
            # Feature selection parameters
            "feature_groups": args.feature_groups,
            
            # Dataset parameters
            "dataset": args.dataset,
            "custom_dataset": args.custom_dataset,
            "hf_only": args.hf_only,
            "custom_only": args.custom_only,
            
            # System info
            "model_type": "GestureClassifier",
            "feature_type": "angle_invariant",
            "device": device.type
        }
        
        # Initialize W&B
        if args.sweep:
            # In sweep mode, wandb.init() gets the hyperparameters from the sweep controller
            if not wandb.run:
                if args.team:
                    wandb.init(project=args.project, entity=args.team)
                else:
                    wandb.init(project=args.project)
            
            # Override args with sweep values
            for key, value in wandb.config.items():
                if key in wandb_config:
                    wandb_config[key] = value  # Update config with sweep values
                    
                    # Also update args for consistency
                    if hasattr(args, key):
                        setattr(args, key, value)
        else:
            # Normal run, use args to configure wandb
            if args.team:
                wandb.init(project=args.project, entity=args.team, name=args.name, config=wandb_config)
            else:
                wandb.init(project=args.project, name=args.name, config=wandb_config)
        
        # Log system info and dependencies
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        print(f" W&B initialized: {wandb.run.name}")
    
    # Load datasets
    hf_dataset = None
    custom_data = None
    
    # Load HuggingFace dataset
    if not args.custom_only:
        print(f"📥 Loading HuggingFace dataset: {args.dataset}")
        start_time = time.time()
        try:
            hf_dataset = load_dataset(args.dataset)
            print(f"✅ HuggingFace dataset loaded in {time.time() - start_time:.2f}s")
            print(f"📊 HuggingFace dataset stats: {len(hf_dataset['train'])} training samples")
        except Exception as e:
            print(f"⚠️ Error loading dataset {args.dataset}: {e}")
            if not args.custom_dataset:
                print("🔄 Falling back to default dataset: Javtor/rock-paper-scissors")
                hf_dataset = load_dataset("Javtor/rock-paper-scissors")
                print(f"✅ Fallback dataset loaded in {time.time() - start_time:.2f}s")
    
    # Load custom dataset
    if args.custom_dataset and not args.hf_only:
        custom_data = load_custom_dataset(args.custom_dataset)
        if not custom_data:
            print("⚠️ No custom data loaded. Continuing with HuggingFace dataset only.")
    
    # Ensure we have at least one dataset
    if not hf_dataset and not custom_data:
        raise ValueError("No dataset available. Please provide either a valid HuggingFace dataset or custom dataset path.")
    
    # Create combined dataset for training
    print("🔗 Creating combined dataset...")
    
    # Get HuggingFace train split if available
    hf_train_split = hf_dataset['train'] if hf_dataset else None
    
    # Split custom data into train and val if we're only using custom data
    custom_train = None
    custom_val = None
    
    if custom_data:
        if args.custom_only:
            # Split custom data into train and validation
            print("🔪 Splitting custom dataset into train/validation sets...")
            
            # Extract labels for stratification
            labels = [item['label'] for item in custom_data]
            
            # Split the indices
            train_indices, val_indices = train_test_split(
                range(len(custom_data)), 
                test_size=0.2, 
                random_state=42,
                stratify=labels
            )
            
            custom_train = [custom_data[i] for i in train_indices]
            custom_val = [custom_data[i] for i in val_indices]
            
            print(f"🔍 Custom split result: {len(custom_train)} train / {len(custom_val)} validation images")
        else:
            # Use all custom data for training
            custom_train = custom_data
    
    # Create training dataset
    full_train_dataset = CombinedHandGestureDataset(
        hf_dataset=hf_train_split,
        custom_data=custom_train
    )
    
    # Create validation dataset
    if args.custom_only and custom_val:
        # Use custom validation split
        val_dataset = CombinedHandGestureDataset(
            hf_dataset=None,
            custom_data=custom_val
        )
    elif hf_train_split:
        # Split HuggingFace training data for validation
        print("🔪 Splitting HuggingFace dataset into train/validation sets...")
        train_val_split = hf_train_split.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
        hf_train_dataset_split = train_val_split['train']
        hf_val_dataset_split = train_val_split['test']
        
        # Recreate training dataset with split data
        full_train_dataset = CombinedHandGestureDataset(
            hf_dataset=hf_train_dataset_split,
            custom_data=custom_train
        )
        
        # Create validation dataset
        val_dataset = CombinedHandGestureDataset(
            hf_dataset=hf_val_dataset_split,
            custom_data=None
        )
        
        print(f"🔍 HuggingFace split result: {len(hf_train_dataset_split)} train / {len(hf_val_dataset_split)} validation images")
    else:
        raise ValueError("Cannot create validation set. Need either HuggingFace dataset or custom_only mode with sufficient data.")
    
    # Get features with caching
    train_features = get_or_compute_features(
        full_train_dataset, "train", batch_size=args.batch_size, 
        feature_groups=feature_groups, custom_path=args.custom_dataset
    )
    
    val_features = get_or_compute_features(
        val_dataset, "val", batch_size=args.batch_size, 
        feature_groups=feature_groups, custom_path=args.custom_dataset
    )
    
    # Get input size based on selected features
    input_size = train_features.shape[1] if train_features.size > 0 else 17
    print(f" Using {input_size} features for model training")
    
    # Set precomputed features on existing datasets
    print("🧩 Setting precomputed features on datasets...")
    full_train_dataset.precomputed_features = train_features
    val_dataset.precomputed_features = val_features
    
    # Create data loaders
    print("🔄 Creating data loaders...")
    train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model with specified architecture
    print(" Initializing model architecture...")
    model = GestureClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=3,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm,
        use_residual=args.residual
    ).to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizer based on args
    print(f" Setting up {args.optimizer} optimizer with lr={args.lr}...")
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Configure learning rate scheduler
    if args.scheduler:
        print(f" Using {args.scheduler} learning rate scheduler")
        if args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, 
                                                           patience=args.step_size//2)
    else:
        scheduler = None
        print(" No learning rate scheduler selected")
    
    # Log model architecture to W&B
    if args.wandb or args.sweep:
        print(" Setting up W&B model watching...")
        wandb.watch(model, criterion=criterion, log="all", log_freq=10)
    
    # Train the model
    use_wandb = args.wandb or args.sweep
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=args.epochs, use_wandb=use_wandb,
        scheduler=scheduler
    )
    
    # Export to ONNX
    model_path = os.path.join(args.export_dir, "gesture_classifier.onnx")
    export_to_onnx(trained_model, input_size=input_size, use_wandb=use_wandb)
    
    # Test on the test set if available (only for HuggingFace dataset)
    if hf_dataset and 'test' in hf_dataset and not args.custom_only:
        print("\n🧪 Evaluating on HuggingFace test set...")
        
        # For test set, use the original HuggingFace approach since no custom data mixing needed
        test_features = get_or_compute_features(
            hf_dataset['test'], "test", batch_size=args.batch_size, 
            feature_groups=feature_groups, custom_path=args.custom_dataset
        )
        
        # Use original HuggingFace dataset class for test set
        test_dataset = HuggingFaceHandGestureDataset(
            hf_dataset['test'], 
            precomputed_features=test_features
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Load best model
        best_model_path = os.path.join(args.export_dir, 'best_gesture_classifier.pth')
        print(f" Loading best model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        correct = 0
        total = 0
        
        start_time = time.time()
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="🔍 Testing model"):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = correct / total
        print(f" Test accuracy: {test_acc:.4f} (evaluated in {time.time() - start_time:.2f}s)")
        
        # Log test accuracy to W&B
        if use_wandb:
            wandb.run.summary["test_accuracy"] = test_acc
    
    # Clean up MediaPipe resources
    print(" Cleaning up resources...")
    hands.close()
    
    # Finish W&B run
    if use_wandb:
        print(" Finalizing W&B run...")
        wandb.finish()
    
    print(" Training pipeline complete!")

if __name__ == "__main__":
    main()
