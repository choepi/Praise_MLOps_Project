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
const emojiDisplay = document.getElementById('emoji-display'); // New: Element to display emoji
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
function extract10Features(landmarks) {
  function dist(p1, p2) {
    return Math.sqrt(
      (p1.x - p2.x) ** 2 +
      (p1.y - p2.y) ** 2 +
      (p1.z - p2.z) ** 2
    );
  }

  function angle(p1, p2, p3) {
    const v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z];
    const v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z];
    const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    const mag1 = Math.sqrt(v1.reduce((sum, v) => sum + v*v, 0));
    const mag2 = Math.sqrt(v2.reduce((sum, v) => sum + v*v, 0));
    const cosAngle = Math.min(1, Math.max(-1, dot / (mag1 * mag2 || 1e-6)));
    return Math.acos(cosAngle) * (180 / Math.PI);
  }

  const MCP = [5, 9, 13, 17];
  const palmCenter = {
    x: MCP.map(i => landmarks[i].x).reduce((a, b) => a + b) / MCP.length,
    y: MCP.map(i => landmarks[i].y).reduce((a, b) => a + b) / MCP.length,
    z: MCP.map(i => landmarks[i].z).reduce((a, b) => a + b) / MCP.length,
  };

  const palmWidth = dist(landmarks[5], landmarks[17]) || 1e-6;

  const features = [
    dist(landmarks[4], palmCenter) / palmWidth,
    dist(landmarks[8], palmCenter) / palmWidth,
    dist(landmarks[12], palmCenter) / palmWidth,
    dist(landmarks[16], palmCenter) / palmWidth,
    dist(landmarks[20], palmCenter) / palmWidth,
    angle(landmarks[2], landmarks[3], landmarks[4]),
    angle(landmarks[6], landmarks[7], landmarks[8]),
    angle(landmarks[10], landmarks[11], landmarks[12]),
    angle(landmarks[14], landmarks[15], landmarks[16]),
    angle(landmarks[18], landmarks[19], landmarks[20])
  ];

  return new Float32Array(features);
}

 
// 5. Game logic: countdown and then play one round (capture frame, run detection & inference, show result)
async function playRound() {
  // Hide feedback from previous round (if any)
  feedbackDiv.style.display = 'none';
 
  // Countdown before starting the game
  await countdown();
 
  // Analysis
  // Draw current frame to canvas
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
 
 
 
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
 
 
 
 
 
 
 
 
 
 // Map prediction to emoji
  const emojiMap = {
    "rock": "✊",
    "paper": "🖐️",
    "scissors": "✌️"
  };


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
    const inputFeatures = extract10Features(landmarks);

    // Prepare ONNX input tensor and run model
    console.log("inputFeatures:", inputFeatures);
    for (let i = 0; i < inputFeatures.length; i++) {
      if (isNaN(inputFeatures[i]) || !isFinite(inputFeatures[i])) {
        console.log("NAN or infinite value found in inputFeatures at index", i);
      }
    }
    const inputTensor = new ort.Tensor('float32', inputFeatures, [1, inputFeatures.length]);
    console.log("inference is running")
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
        playerEmoji = `👑${playerEmoji}`; // Add crown to winner
      } else {
        outcome = "😭 You lose. ";
        computerEmoji = `👑${computerEmoji}`; // Add crown to winner
      }
    }
    // Display result
    // Removed emojiDisplay logic
    // emojiDisplay.textContent = emojiMap[prediction]; // Display the corresponding emoji
    // emojiDisplay.style.display = 'block'; // Make the emoji visible

    resultDiv.innerHTML = `<span class="player-emoji">${playerEmoji}</span> vs <span class="computer-emoji">${computerEmoji}</span><br>${outcome}`;
    resultDiv.classList.add('animate-result'); // Add class for animation

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
    // Check if the user's chosen gesture matches the predicted one
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
  console.log("Initialization complete.");
})();
 
// 9. Set up the Play button to start a round
const playButton = document.getElementById('play-button');
playButton.addEventListener('click', () => {
  playRound();
});
