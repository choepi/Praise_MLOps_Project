// script.js

// Mapping of class indices to gesture names, per training: 0=paper, 1=rock, 2=scissors
const CLASS_NAMES = ["paper", "rock", "scissors"];

// Global state
let ortSession = null;              // ONNX Runtime inference session
let handDetector = null;            // MediaPipe Hands instance
let capturedImage = null;           // Last captured image (for feedback storage)
const collectedData = [];          // Array to store {image: dataURL, label: "rock"/"paper"/"scissors"} for misclassifications

// HTML elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('capture-canvas');
const resultDiv = document.getElementById('result');
const playButton = document.getElementById('play-button');
const feedbackDiv = document.getElementById('feedback');
const feedbackButtons = document.querySelectorAll('.feedback-btn');
const downloadBtn = document.getElementById('download-btn');

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
    ortSession = await ort.InferenceSession.create('model.onnx');
    console.log("ONNX model loaded");
  } catch (e) {
    console.error("Failed to load ONNX model:", e);
  }
}

// 4. Helper function: Given a set of landmarks from MediaPipe (array of {x,y,z}), normalize them like in training
function normalizeLandmarks(landmarks) {
  // Convert landmarks to an array of [x,y] (we ignore z for consistency with training)
  const coords = landmarks.map(lm => [lm.x, lm.y]);
  // Translate so that the wrist (index 0) is at origin (0,0)
  const wrist = coords[0].slice();  // copy of [x0, y0]
  for (let i = 0; i < coords.length; i++) {
    coords[i][0] -= wrist[0];
    coords[i][1] -= wrist[1];
  }
  // Scale so that max distance from origin is 1
  let maxDist = 0;
  for (let [x, y] of coords) {
    const dist = Math.sqrt(x*x + y*y);
    if (dist > maxDist) maxDist = dist;
  }
  if (maxDist > 0) {
    for (let i = 0; i < coords.length; i++) {
      coords[i][0] /= maxDist;
      coords[i][1] /= maxDist;
    }
  }
  // Flatten to one array
  const flatCoords = coords.flat();
  return new Float32Array(flatCoords);
}

// 5. Game logic: play one round (capture frame, run detection & inference, show result)
async function playRound() {
  // Hide feedback from previous round (if any)
  feedbackDiv.style.display = 'none';
  resultDiv.textContent = "Analyzing... 🤖";
  
  // Capture current frame from video into canvas
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // Get image data from canvas (MediaPipe can accept the canvas directly as an image input)
  
  // Run MediaPipe hand detection on this frame
  let landmarks = null;
  try {
    // Use handDetector in static image mode by sending one image and waiting for results
    const results = await handDetector.send({image: canvas});
    // (Note: We call send on the canvas which contains the frame. MediaPipe Hands in JS uses onResults callback if continuous,
    // but it also returns a Promise we can await as shown.)
  } catch (err) {
    console.error("Hand detection failed:", err);
  }
  // MediaPipe Hands doesn't return from send(); instead, we need to use the callback or gather results differently.
  // We'll use the callback approach for simplicity: define a one-time callback to get the landmarks.
}

// We realize that MediaPipe's Hands send() is asynchronous and returns results via the onResults callback. 
// So we should set up a temporary onResults handler that will capture the result for our one frame.
// ... continue inside script.js ...

async function playRound() {
  feedbackDiv.style.display = 'none';
  resultDiv.textContent = "Analyzing... 🤖";
  
  // Draw current frame to canvas
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // Define a one-time callback for hand results
  handDetector.onResults(async (results) => {
    // Remove the callback to avoid interference with future calls
    handDetector.onResults(() => {});  // reset to no-op or you can set null if supported
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      resultDiv.textContent = "No hand detected. Please try again.";
      return;
    }
    // Get the first hand's landmarks
    const landmarks = results.multiHandLandmarks[0];
    // Normalize landmarks to feature vector
    const inputFeatures = normalizeLandmarks(landmarks);
    // Prepare ONNX input tensor and run model
    const inputTensor = new ort.Tensor('float32', inputFeatures, [1, inputFeatures.length]);
    let prediction;
    try {
      const outputMap = await ortSession.run({ input: inputTensor });
      const outputData = outputMap.output.data;  // Float32Array of length 3
      // Get predicted class (index of max value)
      const predIndex = outputData.indexOf(Math.max(...outputData));
      prediction = CLASS_NAMES[predIndex];
    } catch (err) {
      console.error("ONNX inference failed:", err);
      resultDiv.textContent = "Error in model inference.";
      return;
    }
    // Randomly select computer's move
    const compIndex = Math.floor(Math.random() * 3);
    const compMove = CLASS_NAMES[compIndex];
    // Determine winner
    let outcome;
    if (prediction === compMove) {
      outcome = "It's a draw!";
    } else {
      // Define win conditions: rock beats scissors, scissors beats paper, paper beats rock
      if (
        (prediction === "rock" && compMove === "scissors") ||
        (prediction === "scissors" && compMove === "paper") ||
        (prediction === "paper" && compMove === "rock")
      ) {
        outcome = "You win! 🎉";
      } else {
        outcome = "You lose. 😢";
      }
    }
    // Display result
    resultDiv.textContent = `You: ${prediction.toUpperCase()}, Computer: ${compMove.toUpperCase()}. ${outcome}`;
    // Save the current frame image data and predicted label
    capturedImage = canvas.toDataURL("image/png");
    // Show feedback options
    feedbackDiv.style.display = 'block';
  });
  
  // Send the current frame to MediaPipe for processing (this will trigger the onResults above)
  await handDetector.send({ image: canvas });
}


// ... continue inside script.js ...

// 6. Feedback handling: if user says prediction was wrong and chooses correct gesture
feedbackButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const correctGesture = btn.getAttribute('data-gesture');  // "rock", "paper", or "scissors"
    // Determine what the model predicted from resultDiv (we stored 'prediction' internally, but let's parse for safety)
    const resultText = resultDiv.textContent;
    // Only proceed if the user-chosen gesture is different from model's prediction:
    if (resultText && correctGesture && resultText.includes(`You: ${correctGesture.toUpperCase()}`)) {
      // If the prediction was actually correct (user clicked the same as predicted), no correction needed.
      feedbackDiv.style.display = 'none';
      return;
    }
    // Otherwise, store the captured image and correct label
    if (capturedImage) {
      collectedData.push({ image: capturedImage, label: correctGesture });
      console.log("Collected corrected sample: label =", correctGesture);
      // Show download button if not already
      downloadBtn.style.display = 'inline-block';
      // Hide feedback after recording
      feedbackDiv.style.display = 'none';
      alert(`Thanks! Saved this image as '${correctGesture}' for retraining.`);
    }
  });
});

// 7. Download collected data as a JSON file
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

// 8. Initialize everything on page load
(async function init() {
  await setupWebcam();
  setupHandDetector();
  await loadModel();
  // Enable the Play button after everything is ready
  playButton.disabled = false;
  playButton.textContent = "Play Round";
  console.log("Initialization complete.");
})();

// 9. Set up the Play button to start a round
playButton.addEventListener('click', () => {
  playRound();
});
