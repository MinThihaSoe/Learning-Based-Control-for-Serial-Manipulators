import os
import numpy as np
from sympy import symbols, sin, cos, Matrix, pi, simplify
import time
import matplotlib.pyplot as plt

# Load dataset with theta values
script_dir = os.path.dirname(__file__)
IRB120_path = os.path.join(script_dir, 'IRB120', 'robot_inverse_kinematics_dataset.csv')

# Load dataset (assuming the dataset is CSV and has no header)
data = np.loadtxt(IRB120_path, delimiter=',', skiprows=1)
input_data = data[:, :-3]  # features (joint angles or input variables)
target_data = data[:, -3:]  # labels (x, y, z positions)

# Forward Kinematic using Transformation matrices
# Define symbolic variables for theta, link lengths, and DH parameters
t1, t2, t3, t4, t5, t6, t7 = symbols("\\theta_1 \\theta_2 \\theta_3 \\theta_4 \\theta_5 \\theta_6 \\theta_7")
a1, a2, a3, d2, d4, d6 = symbols("a_1 a_2 a_3 d_2 d_4 d_6")
alpha_i1, a_i1, d_i, t_i = symbols("\\alpha_{i-1} a_{i-1} d_i \\theta_i")

# Define transformation matrices
Rx = Matrix([[1, 0, 0, 0],
             [0, cos(alpha_i1), -sin(alpha_i1), 0],
             [0, sin(alpha_i1), cos(alpha_i1), 0],
             [0, 0, 0, 1]])

Dx = Matrix([[1, 0, 0, a_i1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

Rz = Matrix([[cos(t_i), -sin(t_i), 0, 0],
             [sin(t_i), cos(t_i), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

Dz = Matrix([[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, d_i],
             [0, 0, 0, 1]])

T = Rx * Dx * Rz * Dz  # Transformation matrix

# Define modified DH parameters for IRB 120
alpha_vals = Matrix([0, pi/2, 0, -pi/2, pi/2, pi/2, 0])
a_vals = Matrix([0, 0, 0.270, 0.070, 0, 0, 0])
d_vals = Matrix([0, 0, 0, 0.320, 0, 0, 0])
theta_vals = Matrix([t1, t2 + pi/2, t3, t4, t5 - pi, t6, 0])

# Calculate each transformation matrix
T1 = T.subs({alpha_i1: alpha_vals[0], a_i1: a_vals[0], d_i: d_vals[0], t_i: theta_vals[0]})
T2 = T.subs({alpha_i1: alpha_vals[1], a_i1: a_vals[1], d_i: d_vals[1], t_i: theta_vals[1]})
T3 = T.subs({alpha_i1: alpha_vals[2], a_i1: a_vals[2], d_i: d_vals[2], t_i: theta_vals[2]})
T4 = T.subs({alpha_i1: alpha_vals[3], a_i1: a_vals[3], d_i: d_vals[3], t_i: theta_vals[3]})
T5 = T.subs({alpha_i1: alpha_vals[4], a_i1: a_vals[4], d_i: d_vals[4], t_i: theta_vals[4]})
T6 = T.subs({alpha_i1: alpha_vals[5], a_i1: a_vals[5], d_i: d_vals[5], t_i: theta_vals[5]})
T7 = T.subs({alpha_i1: alpha_vals[6], a_i1: a_vals[6], d_i: d_vals[6], t_i: theta_vals[6]})

T_final = T1 * T2 * T3 * T4 * T5 * T6 * T7  # Final transformation matrix
Position = T_final[0:3,3]

# List to store cumulative times
cumulative_times = []

# Total number of data points in the dataset
total_data_points = len(target_data)

# Start tracking total time for the loop
start_time = time.time()

# Loop through the dataset and process each row
for i in range(total_data_points):
    # Get the joint angles for this row
    theta_values_row = data[i, :6]  # Adjust if theta values are in different columns

    # Print current example being processed
    print(f"Processing data point {i+1} out of {total_data_points}...")

    # Create a substitution dictionary for the current theta values
    theta_subs = {t1: theta_values_row[0], t2: theta_values_row[1], t3: theta_values_row[2],
                  t4: theta_values_row[3], t5: theta_values_row[4], t6: theta_values_row[5]}

    # Time each substitution and matrix multiplication
    row_start_time = time.time()

    # Substitute theta values into Position to find xyz
    xyz = Position.subs(theta_subs)

    # Record the time taken for this row
    row_end_time = time.time()

    # Calculate time taken for this row and add to cumulative time
    row_time = row_end_time - row_start_time
    if i == 0:
        cumulative_times.append(row_time)
    else:
        cumulative_times.append(cumulative_times[-1] + row_time)

# Total time for processing all rows
end_time = time.time()
DHruntime= end_time - start_time

import torch
import torch.nn as nn
import torch.optim as optim

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

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)  # input_data should be shaped [batch_size, 6]
    loss = criterion(outputs, target_data)  # target_data should be [batch_size, 3]

    # Backward and optimize
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model and calculate error on test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f'\nTest Loss (MSE): {test_loss.item():.4f}')

# List to store cumulative times
nn_cumulative_times = []
# Track time for Neural Network prediction
nn_start_time = time.time()

# Predict with the trained neural network for the 5000 data points
for i in range(total_data_points):

    # Predict the end-effector position using the neural network
    input_row = input_data[i].clone().detach().to(device)

    nn_row_start_time =  time.time()
    with torch.no_grad():
        nn_position = model(input_row.unsqueeze(0))  # Predict in batches (1 at a time here)
    nn_row_end_time= time.time()
    
    # Calculate the time per prediction
    nn_row_time = nn_row_end_time - nn_row_start_time
    print(f"Processing data point {i+1} out of {total_data_points}...")

    if i == 0:
        nn_cumulative_times.append(nn_row_time)
    else:
        nn_cumulative_times.append(nn_cumulative_times[-1] + nn_row_time)

# Total time for Neural Network predictions
nn_end_time = time.time()

print(f"\nTotal Time for Calculation: {DHruntime:.4f} seconds")
print(f"Total Time for Neural Network Prediction: {nn_end_time - nn_start_time:.4f} seconds")

# Plot cumulative times for both methods with a logarithmic y-axis
plt.plot(range(len(cumulative_times)), cumulative_times, label='D-H Model')
plt.plot(range(len(nn_cumulative_times)), nn_cumulative_times, label='Neural Network Model')
plt.xlabel('Number of Data Points Processed')
plt.ylabel('Cumulative Computation Time (seconds)')
plt.title('Runtime Analysis: Analytical D-H Model vs Neural Networks model in Forward Kinematics')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
