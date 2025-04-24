# Rock-Paper-Scissors via Hand Gestures

An interactive machine learning-powered game that lets users play Rock-Paper-Scissors using real-time hand gesture recognition via webcam input. Designed for browser-based or Raspberry Pi deployment.

## 🎯 Project Goals

- Build a gesture-driven version of Rock-Paper-Scissors using computer vision and machine learning
- Enable intuitive and fun interaction with no mouse or keyboard
- Deployable on web (GitHub Pages) or offline (Raspberry Pi)
- Demonstrate end-to-end ML lifecycle: data → model → inference → feedback → retraining

## 🧠 How It Works

1. **Capture Input**: Live webcam feed using OpenCV
2. **Landmark Detection**: Hand keypoints identified using MediaPipe
3. **Gesture Classification**: Orientation-invariant FNN model or rule-based logic
4. **Game Logic**: Outcome determined based on user vs. computer choice
5. **Output Display**: Visual feedback in real-time UI

## 🛠 Tech Stack

- Python, OpenCV, MediaPipe
- ONNX for model inference
- GitHub Pages for frontend deployment
- Raspberry Pi for offline testing
- wandb for logging and experiment tracking
- Roboflow for labeling
- Google Cloud Storage for data management

## 🧪 Testing & CI/CD

- **Black** linter enforced across development
- **Unit tests** for gesture classification and game logic
- **GitHub Actions** handles automatic testing on all pull requests

## 🚀 Deployment Options

- **Web**: Hosted on GitHub Pages (static, ONNX in-browser inference)
- **Offline**: Runs on Raspberry Pi with local webcam + display

## 🔁 Retraining Strategy

- Misclassified gestures are flagged during gameplay
- Data is labeled via Roboflow and stored in Google Cloud
- Retraining is manually triggered once enough samples are collected

## 🗂 Repository Structure