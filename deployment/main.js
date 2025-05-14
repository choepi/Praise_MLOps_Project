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
const scoreboardDiv = document.getElementById('scoreboard');

// Create upload status element if it doesn't exist
let uploadStatusDiv = document.getElementById('upload-status');
if (!uploadStatusDiv) {
  uploadStatusDiv = document.createElement('div');
  uploadStatusDiv.id = 'upload-status';
  uploadStatusDiv.style.display = 'none';
  // Insert it after the feedback div
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

// Function to update the scoreboard display
function updateScoreboard() {
  if (userScoreSpan && pcScoreSpan) {
    userScoreSpan.textContent = userScore;
    pcScoreSpan.textContent = pcScore;
  } else {
    console.error("Error: Score span elements not found!");
  }
}

// 2. Initialize MediaPipe Hands (for hand landmark detection)
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
  // We will not start continuous detection; instead, we will use handDetector on-demand when the user plays a round.
  console.log("MediaPipe Hands ready");
}

// 3. Load the ONNX model and create an inference session
async function loadModel() {
  try {
    ortSession = await ort.InferenceSession.create('gesture_classifier.onnx');
    console.log("ONNX model loaded");
  } catch (e) {
    console.error("Failed to load ONNX model:", e);
  }
}

// 4. Hand feature extraction function
function extractHandFeatures(landmarks) {
  // Helper function to calculate distance
  function distance(p1, p2) {
    return Math.sqrt(
      Math.pow(p2.x - p1.x, 2) +
      Math.pow(p2.y - p1.y, 2) +
      Math.pow(p2.z - p1.z, 2)
    );
  }

  // Helper function to calculate angle
  function calculateAngle(p1, p2, p3) {
    // Vectors from p2 to p1 and p2 to p3
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

    // Dot product
    const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;

    // Magnitudes
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

    // Angle in radians, then convert to degrees
    const cosAngle = dotProduct / (mag1 * mag2);
    // Clamp to avoid numerical issues
    const clampedCosAngle = Math.max(-1, Math.min(1, cosAngle));
    return Math.acos(clampedCosAngle) * (180 / Math.PI);
  }

  // MediaPipe hand landmark indices
  const WRIST = 0;
  const THUMB_TIP = 4;
  const INDEX_FINGER_TIP = 8;
  const INDEX_FINGER_MCP = 5;  // Base of index finger
  const MIDDLE_FINGER_TIP = 12;
  const MIDDLE_FINGER_MCP = 9;  // Base of middle finger
  const RING_FINGER_TIP = 16;
  const RING_FINGER_MCP = 13;  // Base of ring finger
  const PINKY_TIP = 20;
  const PINKY_MCP = 17;  // Base of pinky finger

  // Calculate palm center (average of base points of all fingers)
  const palmCenter = {
    x: (landmarks[INDEX_FINGER_MCP].x + landmarks[MIDDLE_FINGER_MCP].x +
      landmarks[RING_FINGER_MCP].x + landmarks[PINKY_MCP].x) / 4,
    y: (landmarks[INDEX_FINGER_MCP].y + landmarks[MIDDLE_FINGER_MCP].y +
      landmarks[RING_FINGER_MCP].y + landmarks[PINKY_MCP].y) / 4,
    z: (landmarks[INDEX_FINGER_MCP].z + landmarks[MIDDLE_FINGER_MCP].z +
      landmarks[RING_FINGER_MCP].z + landmarks[PINKY_MCP].z) / 4
  };

  // Calculate palm width (distance between index MCP and pinky MCP)
  const palmWidth = distance(landmarks[INDEX_FINGER_MCP], landmarks[PINKY_MCP]);

  // Feature 1: Finger extension ratios
  const thumbExtension = distance(landmarks[THUMB_TIP], palmCenter) / palmWidth;
  const indexExtension = distance(landmarks[INDEX_FINGER_TIP], palmCenter) / palmWidth;
  const middleExtension = distance(landmarks[MIDDLE_FINGER_TIP], palmCenter) / palmWidth;
  const ringExtension = distance(landmarks[RING_FINGER_TIP], palmCenter) / palmWidth;
  const pinkyExtension = distance(landmarks[PINKY_TIP], palmCenter) / palmWidth;

  // Feature 2: Finger bending angles
  const thumbAngle = calculateAngle(landmarks[2], landmarks[3], landmarks[4]);  // IP joint
  const indexAngle = calculateAngle(landmarks[6], landmarks[7], landmarks[8]);  // PIP joint
  const middleAngle = calculateAngle(landmarks[10], landmarks[11], landmarks[12]);  // PIP joint
  const ringAngle = calculateAngle(landmarks[14], landmarks[15], landmarks[16]);  // PIP joint
  const pinkyAngle = calculateAngle(landmarks[18], landmarks[19], landmarks[20]);  // PIP joint

  // Feature 3: Inter-fingertip distances
  const indexToMiddleDist = distance(landmarks[INDEX_FINGER_TIP], landmarks[MIDDLE_FINGER_TIP]) / palmWidth;
  const middleToRingDist = distance(landmarks[MIDDLE_FINGER_TIP], landmarks[RING_FINGER_TIP]) / palmWidth;
  const ringToPinkyDist = distance(landmarks[RING_FINGER_TIP], landmarks[PINKY_TIP]) / palmWidth;
  const thumbToIndexDist = distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_TIP]) / palmWidth;

  // Feature 4: Thumb opposition
  const thumbToMiddleDist = distance(landmarks[THUMB_TIP], landmarks[MIDDLE_FINGER_TIP]) / palmWidth;
  const thumbToRingDist = distance(landmarks[THUMB_TIP], landmarks[RING_FINGER_TIP]) / palmWidth;
  const thumbToPinkyDist = distance(landmarks[THUMB_TIP], landmarks[PINKY_TIP]) / palmWidth;

  // Return all features as Float32Array (needed for ONNX)
  return new Float32Array([
    // Finger extension ratios
    thumbExtension, indexExtension, middleExtension, ringExtension, pinkyExtension,

    // Finger bending angles
    thumbAngle, indexAngle, middleAngle, ringAngle, pinkyAngle,

    // Inter-fingertip distances
    indexToMiddleDist, middleToRingDist, ringToPinkyDist, thumbToIndexDist,

    // Thumb opposition
    thumbToMiddleDist, thumbToRingDist, thumbToPinkyDist
  ]);
}

// 5. Gesture prediction function
async function predictGesture(landmarks) {
  if (!ortSession || !landmarks) {
    console.error("ONNX session or landmarks not available");
    return null;
  }

  try {
    // Extract features
    const inputFeatures = extractHandFeatures(landmarks);

    // Store features globally for potential upload
    window.lastPredictionFeatures = Array.from(inputFeatures);

    // Log features for debugging
    console.log("Input features:", inputFeatures);

    // Check for NaN or Infinity values
    for (let i = 0; i < inputFeatures.length; i++) {
      if (isNaN(inputFeatures[i]) || !isFinite(inputFeatures[i])) {
        console.log("NAN or infinite value found in inputFeatures at index", i);
      }
    }

    // Prepare input tensor
    const inputTensor = new ort.Tensor('float32', inputFeatures, [1, inputFeatures.length]);

    console.log("Running inference with ONNX model");

    // Run inference
    const outputMap = await ortSession.run({ input: inputTensor });
    const outputData = outputMap.output.data;  // Float32Array containing predictions

    // Get predicted class (index of max value)
    const predIndex = outputData.indexOf(Math.max(...outputData));
    const prediction = CLASS_NAMES[predIndex];
    const confidence = outputData[predIndex];

    console.log("Prediction:", prediction, "Confidence:", confidence);

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

// 6. Helper function to convert data URL to Blob for upload
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

// 7. Cloudinary upload function
async function uploadImageToCloudinary(imageBlob, label) {
  const cloudName = 'dodjfdoxf';
  const uploadPreset = 'praise_mlops';

  // Create form data
  const formData = new FormData();
  formData.append('file', imageBlob);
  formData.append('upload_preset', uploadPreset);
  formData.append('folder', 'user_submissions');

  // Add custom metadata for your ML pipeline
  formData.append('context', `label=${label}`); // You can add more context fields

  try {
    // Make the upload request
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

// 8. Game logic: countdown and then play one round
async function playRound() {
  // Hide feedback from previous round (if any)
  feedbackDiv.style.display = 'none';

  // Reset upload status display
  uploadStatusDiv.style.display = 'none';

  // Countdown before starting the game
  await countdown();

  // Draw current frame to canvas
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Define a one-time callback for hand results
  handDetector.onResults(async (results) => {
    // Remove the callback to avoid interference with future calls
    handDetector.onResults(() => { });  // reset to no-op

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      resultDiv.textContent = "No hand detected. Please try again.";
      return;
    }

    // Get the first hand's landmarks
    const landmarks = results.multiHandLandmarks[0];

    // Use the new prediction function
    const predictionResult = await predictGesture(landmarks);

    if (!predictionResult) {
      resultDiv.textContent = "Error in gesture recognition. Please try again.";
      return;
    }

    const prediction = predictionResult.gesture;

    // Randomly select computer's move
    const compIndex = Math.floor(Math.random() * 3);
    const compMove = CLASS_NAMES[compIndex];

    // Determine winner
    let outcome;
    let playerEmoji = emojiMap[prediction];
    let computerEmoji = emojiMap[compMove];

    if (prediction === compMove) {
      outcome = "🤝 It's a draw!";
    } else {
      // Define win conditions: rock beats scissors, scissors beats paper, paper beats rock
      if (
        (prediction === "rock" && compMove === "scissors") ||
        (prediction === "scissors" && compMove === "paper") ||
        (prediction === "paper" && compMove === "rock")
      ) {
        outcome = "👑 You win! 🎉";
        userScore++;
        playerEmoji = `👑${playerEmoji}`; // Add crown to winner
      } else {
        outcome = "😭 You lose. ";
        pcScore++;
        computerEmoji = `👑${computerEmoji}`; // Add crown to winner
      }
    }

    // Display result
    updateScoreboard();
    resultDiv.innerHTML = `<span class="player-emoji">${playerEmoji}</span> vs <span class="computer-emoji">${computerEmoji}</span><br>${outcome}`;
    resultDiv.classList.add('animate-result'); // Add class for animation

    // Optional: Add confidence display
    resultDiv.innerHTML += `<br><small>Confidence: ${(predictionResult.confidence * 100).toFixed(1)}%</small>`;

    // Save the current frame image data and predicted label
    capturedImage = canvas.toDataURL("image/png");

    // Show feedback options
    feedbackDiv.style.display = 'block';
  });

  // Send the current frame to MediaPipe for processing (this will trigger the onResults above)
  await handDetector.send({ image: canvas });
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

// 9. Updated feedback handling with Cloudinary upload
feedbackButtons.forEach(btn => {
  btn.addEventListener('click', async () => {
    if (isUploading) return; // Prevent multiple clicks during upload

    const correctGesture = btn.getAttribute('data-gesture');  // "rock", "paper", or "scissors"

    // Check if prediction already matches the selected gesture
    const resultText = resultDiv.innerHTML;
    if (resultText && correctGesture && resultText.includes(emojiMap[correctGesture])) {
      // If the prediction was actually correct, no correction needed
      feedbackDiv.style.display = 'none';
      return;
    }

    // Otherwise, process the captured image and correct label
    if (capturedImage) {
      // Store locally for potential download
      collectedData.push({
        image: capturedImage,
        label: correctGesture,
        features: window.lastPredictionFeatures // Store features if available
      });

      // Show download button if not already visible
      downloadBtn.style.display = 'inline-block';

      // Show upload status
      isUploading = true;
      uploadStatusDiv.textContent = "Uploading image to cloud...";
      uploadStatusDiv.style.display = 'block';

      try {
        // Convert data URL to Blob for upload
        const imageBlob = dataURLtoBlob(capturedImage);

        // Upload to Cloudinary
        const uploadResult = await uploadImageToCloudinary(imageBlob, correctGesture);

        if (uploadResult.success) {
          uploadStatusDiv.textContent = "Upload successful!";
          uploadStatusDiv.className = 'success';
          console.log("Image uploaded to Cloudinary:", uploadResult);

          // Optional: Store the Cloudinary URL with your collected data
          collectedData[collectedData.length - 1].cloudinaryUrl = uploadResult.imageUrl;

          setTimeout(() => {
            uploadStatusDiv.style.display = 'none';
            feedbackDiv.style.display = 'none';
          }, 2000); // Hide after 2 seconds
        } else {
          uploadStatusDiv.textContent = "Upload failed. Data saved locally.";
          uploadStatusDiv.className = 'error';
          console.error("Cloudinary upload failed:", uploadResult.error);

          setTimeout(() => {
            uploadStatusDiv.style.display = 'none';
            feedbackDiv.style.display = 'none';
          }, 3000); // Hide after 3 seconds
        }
      } catch (error) {
        console.error("Error in upload process:", error);
        uploadStatusDiv.textContent = "Upload error. Data saved locally.";
        uploadStatusDiv.className = 'error';

        setTimeout(() => {
          uploadStatusDiv.style.display = 'none';
          feedbackDiv.style.display = 'none';
        }, 3000); // Hide after 3 seconds
      } finally {
        isUploading = false;
      }
    }
  });
});

// 10. Updated download function to include features
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

// 11. Initialize everything on page load
(async function init() {
  await setupWebcam();
  setupHandDetector();
  await loadModel();
  updateScoreboard(); // Initialize scoreboard display
  console.log("Initialization complete.");
})();

// 12. Set up the Play button to start a round
const playButton = document.getElementById('play-button');
playButton.addEventListener('click', () => {
  playRound();
});