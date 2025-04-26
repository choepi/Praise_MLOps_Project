import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a simple dummy model
class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.fc = nn.Linear(10, 2)  # 10 inputs → 2 outputs (like rock-paper!)

    def forward(self, x):
        return self.fc(x)

# 2. Create dummy data
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100,))  # 0 or 1 labels

# 3. Initialize model, loss, optimizer
model = DummyNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train (only a few epochs, fast)
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print("✅ Training finished!")

# 5. Export the model to ONNX
dummy_input = torch.randn(1, 10)  # Shape matches model input
torch.onnx.export(
    model,
    dummy_input,
    "deployment/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print("✅ model.onnx exported successfully!")