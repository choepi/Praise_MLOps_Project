async function runInference(imageData) {
    const session = await ort.InferenceSession.create('./model.onnx');
    
    // You would pre-process the image to match model input here
    const tensor = new ort.Tensor('float32', imageData, [1, 3, 224, 224]);
    
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    
    const output = results.output.data;
    document.getElementById("result").innerText = "Prediction: " + output;
}

document.getElementById("upload").addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        // TODO: Convert image to tensor here...
        alert("Image selected! Inference not yet implemented."); 
        // Placeholder
    }
});