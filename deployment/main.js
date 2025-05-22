// Mapping of class indices to gesture names, per training: 0=paper, 1=rock, 2=scissors
const CLASS_NAMES = ["paper", "rock", "scissors"];

// Map prediction to emoji
const emojiMap = {
  "rock": "✊",
  "paper": "🖐️",
  "scissors": "✌️"
};

// Global state
let ortSession = null;              // ONNX Runtime inference session
let handDetector = null;            // MediaPipe Hands instance
let handDetectionModel = null;      // Hand detection model
let capturedImage = null;           // Last captured image (for feedback storage)
const collectedData = [];           // Array to store data for misclassifications
let isUploading = false;            // Flag to prevent multiple uploads
window.lastPredictionFeatures = null; // Store last features for upload
let userScore = 0;
let pcScore = 0;

// HTML elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('capture-canvas');
const resultDiv = document.getElementById('result');
const feedbackDiv = document.getElementById('feedback');
const feedbackButtons = document.querySelectorAll('.feedback-btn');
const downloadBtn = document.getElementById('download-btn');
const userScoreSpan = document.getElementById('user-score');
const pcScoreSpan = document.getElementById('pc-score');

// Create display canvas for cropped hand
let cropDisplayCanvas = document.getElementById('crop-display');
if (!cropDisplayCanvas) {
  cropDisplayCanvas = document.createElement('canvas');
  cropDisplayCanvas.id = 'crop-display';
  cropDisplayCanvas.className = 'crop-display';
  cropDisplayCanvas.style.display = 'none';
  
  // Position it relative to the video container
  const videoContainer = document.querySelector('.video-container') || document.body;
  if (videoContainer === document.body) {
    // Create video container if it doesn't exist
    const container = document.createElement('div');
    container.className = 'video-container';
    container.style.position = 'relative';
    container.style.display = 'inline-block';
    video.parentNode.insertBefore(container, video);
    container.appendChild(video);
    container.appendChild(cropDisplayCanvas);
  } else {
    videoContainer.appendChild(cropDisplayCanvas);
  }
}

// Create upload status element if it doesn't exist
let uploadStatusDiv = document.getElementById('upload-status');
if (!uploadStatusDiv) {
  uploadStatusDiv = document.createElement('div');
  uploadStatusDiv.id = 'upload-status';
  uploadStatusDiv.style.display = 'none';
  feedbackDiv.parentNode.insertBefore(uploadStatusDiv, feedbackDiv.nextSibling);
}

// 1. Initialize webcam video stream
async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await new Promise((resolve) => (video.onloadedmetadata = resolve));
    console.log("Webcam video started");
  } catch (err) {
    alert("Error accessing webcam: " + err);
  }
}

// 2. Load hand detection model
async function loadHandDetectionModel() {
  try {
    // Using MediaPipe Hands for detection, then cropping
    handDetectionModel = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    handDetectionModel.setOptions({
      maxNumHands: 1,
      modelComplexity: 0, // Fastest for detection
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    console.log("Hand detection model loaded");
    return true;
  } catch (error) {
    console.error("Failed to load hand detection model:", error);
    return false;
  }
}

// 3. Detect hand and get bounding box
async function detectHand(sourceCanvas) {
  return new Promise((resolve) => {
    if (!handDetectionModel) {
      resolve({ detected: false });
      return;
    }

    handDetectionModel.onResults((results) => {
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        
        // Calculate bounding box from landmarks
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        
        landmarks.forEach(landmark => {
          minX = Math.min(minX, landmark.x);
          minY = Math.min(minY, landmark.y);
          maxX = Math.max(maxX, landmark.x);
          maxY = Math.max(maxY, landmark.y);
        });
        
        // Convert to pixel coordinates and add padding
        const padding = 0.1; // 10% padding
        const width = maxX - minX;
        const height = maxY - minY;
        
        const cropX = Math.max(0, (minX - padding) * sourceCanvas.width);
        const cropY = Math.max(0, (minY - padding) * sourceCanvas.height);
        const cropWidth = Math.min(sourceCanvas.width - cropX, (width + padding * 2) * sourceCanvas.width);
        const cropHeight = Math.min(sourceCanvas.height - cropY, (height + padding * 2) * sourceCanvas.height);
        
        resolve({
          detected: true,
          x: cropX,
          y: cropY,
          width: cropWidth,
          height: cropHeight
        });
      } else {
        resolve({ detected: false });
      }
    });
    
    handDetectionModel.send({ image: sourceCanvas });
  });
}

// 4. Crop hand region and display it
function cropAndDisplayHand(sourceCanvas, detection) {
  if (!detection.detected) {
    cropDisplayCanvas.style.display = 'none';
    return sourceCanvas;
  }
  
  // Create cropped canvas
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = detection.width;
  cropCanvas.height = detection.height;
  const cropCtx = cropCanvas.getContext('2d');
  
  // Draw cropped region
  cropCtx.drawImage(
    sourceCanvas,
    detection.x, detection.y, detection.width, detection.height,
    0, 0, detection.width, detection.height
  );
  
  // Display the cropped hand
  const displaySize = 150;
  const aspectRatio = detection.width / detection.height;
  const displayWidth = aspectRatio > 1 ? displaySize : displaySize * aspectRatio;
  const displayHeight = aspectRatio > 1 ? displaySize / aspectRatio : displaySize;
  
  cropDisplayCanvas.width = displayWidth;
  cropDisplayCanvas.height = displayHeight;
  cropDisplayCanvas.style.width = displayWidth + 'px';
  cropDisplayCanvas.style.height = displayHeight + 'px';
  
  const displayCtx = cropDisplayCanvas.getContext('2d');
  displayCtx.drawImage(cropCanvas, 0, 0, displayWidth, displayHeight);
  
  cropDisplayCanvas.style.display = 'block';
  
  console.log(`Hand detected and cropped: ${detection.width}x${detection.height} at (${detection.x}, ${detection.y})`);
  return cropCanvas;
}

// 5. Initialize MediaPipe Hands for gesture recognition
function setupHandDetector() {
  handDetector = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  handDetector.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  console.log("MediaPipe Hands ready for gesture recognition");
}

// 6. Load the ONNX model and create an inference session
async function loadModel() {
  try {
    ortSession = await ort.InferenceSession.create('gesture_classifier.onnx');
    console.log("ONNX model loaded");
  } catch (e) {
    console.error("Failed to load ONNX model:", e);
  }
}

// 7. Hand feature extraction function
function extractHandFeatures(landmarks) {
  function distance(p1, p2) {
    return Math.sqrt(
      Math.pow(p2.x - p1.x, 2) +
      Math.pow(p2.y - p1.y, 2) +
      Math.pow(p2.z - p1.z, 2)
    );
  }

  function calculateAngle(p1, p2, p3) {
    const v1 = {
      x: p1.x - p2.x,
      y: p1.y - p2.y,
      z: p1.z - p2.z
    };

    const v2 = {
      x: p3.x - p2.x,
      y: p3.y - p2.y,
      z: p3.z - p2.z
    };

    const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

    const cosAngle = dotProduct / (mag1 * mag2);
    const clampedCosAngle = Math.max(-1, Math.min(1, cosAngle));
    return Math.acos(clampedCosAngle) * (180 / Math.PI);
  }

  const WRIST = 0;
  const THUMB_TIP = 4;
  const INDEX_FINGER_TIP = 8;
  const INDEX_FINGER_MCP = 5;
  const MIDDLE_FINGER_TIP = 12;
  const MIDDLE_FINGER_MCP = 9;
  const RING_FINGER_TIP = 16;
  const RING_FINGER_MCP = 13;
  const PINKY_TIP = 20;
  const PINKY_MCP = 17;

  const palmCenter = {
    x: (landmarks[INDEX_FINGER_MCP].x + landmarks[MIDDLE_FINGER_MCP].x +
      landmarks[RING_FINGER_MCP].x + landmarks[PINKY_MCP].x) / 4,
    y: (landmarks[INDEX_FINGER_MCP].y + landmarks[MIDDLE_FINGER_MCP].y +
      landmarks[RING_FINGER_MCP].y + landmarks[PINKY_MCP].y) / 4,
    z: (landmarks[INDEX_FINGER_MCP].z + landmarks[MIDDLE_FINGER_MCP].z +
      landmarks[RING_FINGER_MCP].z + landmarks[PINKY_MCP].z) / 4
  };

  const palmWidth = distance(landmarks[INDEX_FINGER_MCP], landmarks[PINKY_MCP]);

  const thumbExtension = distance(landmarks[THUMB_TIP], palmCenter) / palmWidth;
  const indexExtension = distance(landmarks[INDEX_FINGER_TIP], palmCenter) / palmWidth;
  const middleExtension = distance(landmarks[MIDDLE_FINGER_TIP], palmCenter) / palmWidth;
  const ringExtension = distance(landmarks[RING_FINGER_TIP], palmCenter) / palmWidth;
  const pinkyExtension = distance(landmarks[PINKY_TIP], palmCenter) / palmWidth;

  const thumbAngle = calculateAngle(landmarks[2], landmarks[3], landmarks[4]);
  const indexAngle = calculateAngle(landmarks[6], landmarks[7], landmarks[8]);
  const middleAngle = calculateAngle(landmarks[10], landmarks[11], landmarks[12]);
  const ringAngle = calculateAngle(landmarks[14], landmarks[15], landmarks[16]);
  const pinkyAngle = calculateAngle(landmarks[18], landmarks[19], landmarks[20]);

  const indexToMiddleDist = distance(landmarks[INDEX_FINGER_TIP], landmarks[MIDDLE_FINGER_TIP]) / palmWidth;
  const middleToRingDist = distance(landmarks[MIDDLE_FINGER_TIP], landmarks[RING_FINGER_TIP]) / palmWidth;
  const ringToPinkyDist = distance(landmarks[RING_FINGER_TIP], landmarks[PINKY_TIP]) / palmWidth;
  const thumbToIndexDist = distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_TIP]) / palmWidth;

  const thumbToMiddleDist = distance(landmarks[THUMB_TIP], landmarks[MIDDLE_FINGER_TIP]) / palmWidth;
  const thumbToRingDist = distance(landmarks[THUMB_TIP], landmarks[RING_FINGER_TIP]) / palmWidth;
  const thumbToPinkyDist = distance(landmarks[THUMB_TIP], landmarks[PINKY_TIP]) / palmWidth;

  return new Float32Array([
    thumbExtension, indexExtension, middleExtension, ringExtension, pinkyExtension,
    thumbAngle, indexAngle, middleAngle, ringAngle, pinkyAngle,
    indexToMiddleDist, middleToRingDist, ringToPinkyDist, thumbToIndexDist,
    thumbToMiddleDist, thumbToRingDist, thumbToPinkyDist
  ]);
}

// 8. Gesture prediction function
async function predictGesture(landmarks) {
  if (!ortSession || !landmarks) {
    console.error("ONNX session or landmarks not available");
    return null;
  }

  try {
    const inputFeatures = extractHandFeatures(landmarks);
    window.lastPredictionFeatures = Array.from(inputFeatures);

    const inputTensor = new ort.Tensor('float32', inputFeatures, [1, inputFeatures.length]);
    const outputMap = await ortSession.run({ input: inputTensor });
    const outputData = outputMap.output.data;

    const predIndex = outputData.indexOf(Math.max(...outputData));
    const prediction = CLASS_NAMES[predIndex];
    const confidence = outputData[predIndex];

    return {
      gesture: prediction,
      confidence: confidence,
      allProbabilities: outputData
    };
  } catch (err) {
    console.error("ONNX inference failed:", err);
    return null;
  }
}

// 9. Function to update the scoreboard display
function updateScoreboard() {
  if (userScoreSpan && pcScoreSpan) {
    userScoreSpan.textContent = userScore;
    pcScoreSpan.textContent = pcScore;
  }
}

// 10. Main game logic
async function playRound() {
  feedbackDiv.style.display = 'none';
  uploadStatusDiv.style.display = 'none';
  cropDisplayCanvas.style.display = 'none';

  await countdown();

  // Capture frame
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Try to detect hand
  resultDiv.innerHTML = '<span style="color: #007aff;">🔍 Detecting hand...</span>';
  
  const detection = await detectHand(canvas);
  const imageToProcess = cropAndDisplayHand(canvas, detection);
  
  if (detection.detected) {
    resultDiv.innerHTML = '<span style="color: #007aff;">✋ Hand found! Analyzing gesture...</span>';
  } else {
    resultDiv.innerHTML = '<span style="color: #ff9500;">⚠️ Using full frame...</span>';
  }

  await new Promise(resolve => setTimeout(resolve, 500));

  // Process with MediaPipe for gesture recognition
  handDetector.onResults(async (results) => {
    handDetector.onResults(() => {});

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      resultDiv.innerHTML = '<span style="color: #ff3b30;">❌ No hand landmarks detected. Please try again.</span>';
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const predictionResult = await predictGesture(landmarks);

    if (!predictionResult) {
      resultDiv.textContent = "Error in gesture recognition. Please try again.";
      return;
    }

    const prediction = predictionResult.gesture;
    const compIndex = Math.floor(Math.random() * 3);
    const compMove = CLASS_NAMES[compIndex];

    let outcome;
    let playerEmoji = emojiMap[prediction];
    let computerEmoji = emojiMap[compMove];

    if (prediction === compMove) {
      outcome = "🤝 It's a draw!";
    } else {
      if (
        (prediction === "rock" && compMove === "scissors") ||
        (prediction === "scissors" && compMove === "paper") ||
        (prediction === "paper" && compMove === "rock")
      ) {
        outcome = "👑 You win! 🎉";
        userScore++;
        playerEmoji = `👑${playerEmoji}`;
      } else {
        outcome = "😭 You lose. ";
        pcScore++;
        computerEmoji = `👑${computerEmoji}`;
      }
    }

    updateScoreboard();
    resultDiv.innerHTML = `<span class="player-emoji">${playerEmoji}</span> vs <span class="computer-emoji">${computerEmoji}</span><br>${outcome}`;
    resultDiv.classList.add('animate-result');

    let infoText = `<br><small>Gesture Confidence: ${(predictionResult.confidence * 100).toFixed(1)}%</small>`;
    resultDiv.innerHTML += infoText;

    capturedImage = canvas.toDataURL("image/png");
    feedbackDiv.style.display = 'block';
  });

  await handDetector.send({ image: imageToProcess });
}

// Countdown helper function
async function countdown() {
  return new Promise((resolve) => {
    let count = 3;
    resultDiv.innerHTML = `<span class="countdown">${count}</span>`;
    const interval = setInterval(() => {
      count--;
      if (count > 0) {
        resultDiv.innerHTML = `<span class="countdown">${count}</span>`;
      } else {
        resultDiv.innerHTML = `<span class="countdown">Go!</span>`;
        clearInterval(interval);
        setTimeout(() => {
          resolve();
        }, 1000);
      }
    }, 1000);
  });
}

// Helper function to convert data URL to Blob for upload
function dataURLtoBlob(dataURL) {
  const arr = dataURL.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);

  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }

  return new Blob([u8arr], { type: mime });
}

// Cloudinary upload function
async function uploadImageToCloudinary(imageBlob, label) {
  const cloudName = 'dodjfdoxf';
  const uploadPreset = 'praise_mlops';

  const formData = new FormData();
  formData.append('file', imageBlob);
  formData.append('upload_preset', uploadPreset);
  formData.append('folder', 'user_submissions');
  formData.append('context', `label=${label}`);

  try {
    const response = await fetch(`https://api.cloudinary.com/v1_1/${cloudName}/image/upload`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    const result = await response.json();

    return {
      success: true,
      imageUrl: result.secure_url,
      publicId: result.public_id,
      metadata: result.context
    };
  } catch (error) {
    console.error('Upload error:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Feedback handling with Cloudinary upload
feedbackButtons.forEach(btn => {
  btn.addEventListener('click', async () => {
    if (isUploading) return;

    const correctGesture = btn.getAttribute('data-gesture');
    const resultText = resultDiv.innerHTML;
    
    if (resultText && correctGesture && resultText.includes(emojiMap[correctGesture])) {
      feedbackDiv.style.display = 'none';
      return;
    }

    if (capturedImage) {
      collectedData.push({
        image: capturedImage,
        label: correctGesture,
        features: window.lastPredictionFeatures
      });

      downloadBtn.style.display = 'inline-block';

      isUploading = true;
      uploadStatusDiv.textContent = "Uploading image to cloud...";
      uploadStatusDiv.style.display = 'block';

      try {
        const imageBlob = dataURLtoBlob(capturedImage);
        const uploadResult = await uploadImageToCloudinary(imageBlob, correctGesture);

        if (uploadResult.success) {
          uploadStatusDiv.textContent = "Upload successful!";
          uploadStatusDiv.className = 'success';
          collectedData[collectedData.length - 1].cloudinaryUrl = uploadResult.imageUrl;

          setTimeout(() => {
            uploadStatusDiv.style.display = 'none';
            feedbackDiv.style.display = 'none';
          }, 2000);
        } else {
          uploadStatusDiv.textContent = "Upload failed. Data saved locally.";
          uploadStatusDiv.className = 'error';

          setTimeout(() => {
            uploadStatusDiv.style.display = 'none';
            feedbackDiv.style.display = 'none';
          }, 3000);
        }
      } catch (error) {
        console.error("Error in upload process:", error);
        uploadStatusDiv.textContent = "Upload error. Data saved locally.";
        uploadStatusDiv.className = 'error';

        setTimeout(() => {
          uploadStatusDiv.style.display = 'none';
          feedbackDiv.style.display = 'none';
        }, 3000);
      } finally {
        isUploading = false;
      }
    }
  });
});

// Download function
downloadBtn.addEventListener('click', () => {
  if (collectedData.length === 0) return;
  const dataStr = JSON.stringify(collectedData, null, 2);
  const blob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "rps_corrections.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// Initialize everything on page load
(async function init() {
  await setupWebcam();
  await loadHandDetectionModel();
  setupHandDetector();
  await loadModel();
  updateScoreboard();
  console.log("Initialization complete.");
})();

// Set up the Play button
const playButton = document.getElementById('play-button');
playButton.addEventListener('click', () => {
  playRound();
});

// Privacy popup logic
window.addEventListener('DOMContentLoaded', () => {
  const popup = document.getElementById('privacy-popup');
  const dismissBtn = document.getElementById('dismiss-popup');

  if (!sessionStorage.getItem('privacyConsent')) {
    popup.style.display = 'flex';
  }

  dismissBtn.addEventListener('click', () => {
    sessionStorage.setItem('privacyConsent', 'true');
    popup.style.display = 'none';
  });
});