// script.js

const CLASS_NAMES = ["paper", "rock", "scissors"];

let ortSession = null;
let handDetector = null;
let capturedImage = null;
const collectedData = [];

const video = document.getElementById('webcam');
const canvas = document.getElementById('capture-canvas');
const resultDiv = document.getElementById('result');
const feedbackDiv = document.getElementById('feedback');
const feedbackButtons = document.querySelectorAll('.feedback-btn');
const downloadBtn = document.getElementById('download-btn');

// --- Init webcam ---
async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(resolve => video.onloadedmetadata = resolve);
  console.log("✅ Webcam initialized");
}

// --- Init hand detector ---
function setupHandDetector() {
  handDetector = new Hands({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  handDetector.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  handDetector.onResults(onHandResults);
  console.log("✅ MediaPipe Hands ready");
}

// --- Load ONNX model ---
async function loadModel() {
  ortSession = await ort.InferenceSession.create('model.onnx');
  console.log("✅ ONNX model loaded");
}

// --- Feature extractor (same logic as train.py) ---
function extract10Features(landmarks) {
  function dist(p1, p2) {
    return Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2);
  }
  function angle(p1, p2, p3) {
    const v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z];
    const v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z];
    const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    const mag1 = Math.sqrt(v1.reduce((s,v) => s + v*v, 0));
    const mag2 = Math.sqrt(v2.reduce((s,v) => s + v*v, 0));
    return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2 || 1e-6)))) * 180 / Math.PI;
  }

  const MCP = [5, 9, 13, 17];
  const palmCenter = {
    x: MCP.map(i => landmarks[i].x).reduce((a,b)=>a+b)/4,
    y: MCP.map(i => landmarks[i].y).reduce((a,b)=>a+b)/4,
    z: MCP.map(i => landmarks[i].z).reduce((a,b)=>a+b)/4
  };
  const palmWidth = dist(landmarks[5], landmarks[17]) || 1e-6;

  return new Float32Array([
    dist(landmarks[4], palmCenter)/palmWidth,
    dist(landmarks[8], palmCenter)/palmWidth,
    dist(landmarks[12], palmCenter)/palmWidth,
    dist(landmarks[16], palmCenter)/palmWidth,
    dist(landmarks[20], palmCenter)/palmWidth,
    angle(landmarks[2], landmarks[3], landmarks[4]),
    angle(landmarks[6], landmarks[7], landmarks[8]),
    angle(landmarks[10], landmarks[11], landmarks[12]),
    angle(landmarks[14], landmarks[15], landmarks[16]),
    angle(landmarks[18], landmarks[19], landmarks[20]),
  ]);
}

// --- Inference pipeline ---
async function onHandResults(results) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    resultDiv.textContent = "🛑 No hand detected.";
    return;
  }

  const landmarks = results.multiHandLandmarks[0];
  drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
  drawLandmarks(ctx, landmarks, { color: '#FF0000', radius: 3 });

  const inputFeatures = extract10Features(landmarks);
  const inputTensor = new ort.Tensor('float32', inputFeatures, [1, inputFeatures.length]);

  try {
    const output = await ortSession.run({ input: inputTensor });
    const predIndex = output.output.data.indexOf(Math.max(...output.output.data));
    const prediction = CLASS_NAMES[predIndex];
    const compMove = CLASS_NAMES[Math.floor(Math.random() * 3)];

    const emojiMap = { rock: "✊", paper: "🖐️", scissors: "✌️" };
    let playerEmoji = emojiMap[prediction];
    let computerEmoji = emojiMap[compMove];
    let outcome;

    if (prediction === compMove) outcome = "🤝 It's a draw!";
    else if (
      (prediction === "rock" && compMove === "scissors") ||
      (prediction === "scissors" && compMove === "paper") ||
      (prediction === "paper" && compMove === "rock")
    ) {
      outcome = "👑 You win!";
      playerEmoji = `👑${playerEmoji}`;
    } else {
      outcome = "😭 You lose.";
      computerEmoji = `👑${computerEmoji}`;
    }

    resultDiv.innerHTML = `<span class="player-emoji">${playerEmoji}</span> vs <span class="computer-emoji">${computerEmoji}</span><br>${outcome}`;
    feedbackDiv.style.display = 'block';
    capturedImage = canvas.toDataURL("image/png");

  } catch (e) {
    resultDiv.textContent = "❌ Error during inference.";
    console.error(e);
  }
}

// --- Trigger a round ---
async function playRound() {
  feedbackDiv.style.display = 'none';
  resultDiv.innerHTML = "⏳ Analyzing...";
  await handDetector.send({ image: video });
}

// --- Countdown (optional but left out for now) ---

// --- Feedback buttons ---
feedbackButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const correct = btn.getAttribute('data-gesture');
    if (capturedImage) {
      collectedData.push({ image: capturedImage, label: correct });
      downloadBtn.style.display = 'inline-block';
      feedbackDiv.style.display = 'none';
      alert(`✅ Sample saved as '${correct}'`);
    }
  });
});

// --- Download button ---
downloadBtn.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(collectedData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "rps_corrections.json";
  a.click();
  URL.revokeObjectURL(url);
});

// --- Init everything ---
(async () => {
  await setupWebcam();
  setupHandDetector();
  await loadModel();
  console.log("✅ System initialized");
})();

// --- Bind play button ---
document.getElementById('play-button').addEventListener('click', playRound);
