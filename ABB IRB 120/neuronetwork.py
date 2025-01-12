import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Get the current working directory (robust path handling)
script_dir = os.path.dirname(__file__)
IRB2400_path = os.path.join(script_dir, 'IRB120', 'robot_inverse_kinematics_dataset.csv')

# Load dataset (assuming the dataset is CSV and has no header)
data = np.loadtxt(IRB2400_path, delimiter=',', skiprows=1)
input_data = data[:, :-3]  # features (joint angles or input variables)
target_data = data[:, -3:]  # labels (x, y, z positions)

# Convert data to PyTorch tensors
input_data = torch.tensor(input_data, dtype=torch.float32)
target_data = torch.tensor(target_data, dtype=torch.float32)

# Split data into train and test sets (80-20 split)
split_idx = int(len(data) * 0.8)
X_train, y_train = input_data[:split_idx], target_data[:split_idx]
X_test, y_test = input_data[split_idx:], target_data[split_idx:]

# Define the neural network
class ManipulatorNet6DoF(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, output_size=3):
        super(ManipulatorNet6DoF, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Parameters
input_size = 6     # Six joint angles as input
hidden_size = 256  # Increased hidden size to capture complexity
output_size = 3    # Output x, y, z position for end-effector position

# Initialize model, loss, and optimizer
model = ManipulatorNet6DoF(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_data = input_data.to(device)
target_data = target_data.to(device)

num_epochs = 600
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)  # input_data should be shaped [batch_size, 6]
    loss = criterion(outputs, target_data)  # target_data should be [batch_size, 3]

    # Backward and optimize
    loss.backward()
    optimizer.step()

# Test the model and calculate error on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        test_losses.append(test_loss.item())

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot Test Loss (MSE) over Epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Test Loss (MSE)')
plt.title('Test Loss (MSE) over Epochs')
plt.legend()
plt.grid()
plt.show()

# Convert predictions and targets to CPU for plotting
test_predictions = test_predictions.cpu().numpy()
y_test = y_test.cpu().numpy()

# Plot predicted vs actual positions for each coordinate (x, y, z)
fig, axes = plt.subplots(3, 1, figsize=(18, 5))
coordinates = ['X', 'Y', 'Z']
for i, ax in enumerate(axes):
    ax.plot(y_test[:, i],y_test[:, i], label='Actual', color='blue', alpha=0.7)
    ax.scatter(y_test[:, i],test_predictions[:, i], label='Predicted', color='orange', alpha=0.7)
    ax.set_title(f'{coordinates[i]} Position')
    ax.set_xlabel('Test Sample')
    ax.set_ylabel('Position')
    ax.legend()

plt.suptitle('Actual vs. Predicted End-Effector Positions')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for main title

plt.show()
