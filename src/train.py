# train.py
import numpy as np
import torch
import torch.nn as nn

# 1. Load preprocessed landmark data
data = np.load("landmarks_data.npz")
X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.long)
X_test  = torch.tensor(data["X_test"], dtype=torch.float32)
y_test  = torch.tensor(data["y_test"], dtype=torch.long)

n_features = X_train.shape[1]  # should be 42
n_classes = 3                  # rock, paper, scissors

# 2. Define the neural network model
class RPSClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, output_dim=3):
        super(RPSClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = RPSClassifier(input_dim=n_features, output_dim=n_classes)

# 3. Set up training components
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
epochs = 20
for epoch in range(1, epochs+1):
    # Forward pass
    logits = model(X_train)
    loss = criterion(logits, y_train)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Compute training accuracy for monitoring
    preds = logits.argmax(dim=1)
    train_acc = (preds == y_train).float().mean().item()
    if epoch % 5 == 0 or epoch == epochs:
        test_logits = model(X_test)
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test).float().mean().item()
        print(f"Epoch {epoch:02d}: loss={loss.item():.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# 5. Save the trained model weights (optional, if we want to keep a PyTorch copy)
torch.save(model.state_dict(), "rps_model.pth")

# 6. Export the model to ONNX format
# Prepare an example input tensor (batch size 1, 42 features)
dummy_input = torch.randn(1, n_features, requires_grad=False)
onnx_path = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, 
                  input_names=["input"], output_names=["output"], 
                  dynamic_axes={"input": [0], "output": [0]})
print(f"Exported the model to {onnx_path}")
