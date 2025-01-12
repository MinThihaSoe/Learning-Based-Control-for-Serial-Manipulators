import os
import numpy as np
import matplotlib.pyplot as plt

# Get the current working directory (robust path handling)
script_dir = os.path.dirname(__file__)
IRB2400_path = os.path.join(script_dir, 'IRB120', 'robot_inverse_kinematics_dataset.csv')

# Load dataset (assuming the dataset is CSV and has no header)
data = np.loadtxt(IRB2400_path, delimiter=',', skiprows=1)
X = data[:, :-3]  # features (joint angles or input variables)
y = data[:, -3:]  # labels (x, y, z positions)

# Initializing variables
m, n = X.shape            # m: number of examples, n: number of features
theta = np.zeros((n, 3))  # Initialize parameter vector for each output (x, y, z)
lambd = 0.01              # The regularization constant (lambda)

# Other variable parameters
alpha = 0.0001          # Learning rate
iterations = 2000        # Number of iterations
batch_size = 200        # Size of the batch 

# Split the data into 80-20
splitset = int(len(data) * 0.8)
X_train = X[:splitset]
y_train = y[:splitset]
X_test = X[splitset:]
y_test = y[splitset:]
print (X_test)
# Sigmoid function
def sigmoid(z):
    # Initialize output array
    s = np.zeros_like(z)
    
    # Apply the sigmoid function depending on signs
    positive_mask = z >= 0
    negative_mask = z < 0
    
    s[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))                        # For non-negative z
    s[negative_mask] = np.exp(z[negative_mask]) / (1 + np.exp(z[negative_mask]))  # For negative z

    return s

# Cost function for multivariate output using Mean Squared Error
def cost_function(X, y, theta, lambd):
    """Computes the cost for multivariate linear regression with L2 regularization."""
    N = len(y)  # Number of examples
    predictions = X @ theta  # Predictions (linear hypothesis)

    # Compute the cost for each output dimension
    cost = np.zeros(theta.shape[1])  # Initialize a vector to hold the cost for each output
    for i in range(theta.shape[1]):  # Loop through each output
        cost[i] = (1 / (2*N)) * np.sum((predictions[:,i] - y[:, i]) ** 2) + (lambd / (2 * N)) * np.sum(np.square(theta[:, i]))
    return cost

# Gradient Descent function
def gradient_descent(X, y, theta, lambd, alpha, iterations):
    """Performs gradient descent to learn theta."""
    N = len(y)  # Number of examples
    cost_history = np.zeros((iterations, theta.shape[1]))  # Initialize cost history as a matrix
    
    for i in range(iterations):
        index = np.random.permutation(N)
        X_shuffled = X[index]
        y_shuffled = y[index]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            batch_size_current = len(y_batch)

            predictions = X_batch @ theta        # Predictions
            gradient = (1 / batch_size_current) * (X_batch.T @ (predictions - y_batch))+ (lambd * theta)  # Gradient calculation

            # Regularization update
            for j in range(theta.shape[1]):  # Loop through each output
                theta[:, j] -= alpha * gradient[:, j]    # Update theta for each output

            current_cost = cost_function(X, y, theta, lambd)  # Compute cost for all outputs
            cost_history[i] = current_cost  # Store the current cost in the matrix

            if i % 1 == 0:
                print(f'Iteration {i}, Cost: {current_cost}')  # This will print a vector
    
    return theta, cost_history  # Return the final theta and the cost history as a matrix

# Perform Gradient Descent
theta, cost_history = gradient_descent(X_train, y_train, theta, lambd, alpha, iterations)

# Plot cost function over iterations
plt.figure(figsize=(12, 6))

# Plot costs for each output variable (x, y, z)
plt.plot(range(len(cost_history)), cost_history[:, 0], 'r', label='Cost for x')
plt.plot(range(len(cost_history)), cost_history[:, 1], 'g', label='Cost for y')
plt.plot(range(len(cost_history)), cost_history[:, 2], 'b', label='Cost for z')

plt.title('Cost Function over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()            # Add legend to distinguish between x, y, z costs
plt.grid()              # Optional: Add grid for better visibility
plt.show()

# Final learned theta values
print("Learned theta parameters: ", theta)

# Make predictions on the test set
y_pred = X_test @ theta

# Plot actual vs predicted values for each output variable
plt.figure(figsize=(12, 6))

# Plot actual vs predicted for x
plt.subplot(3, 1, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0], color='r', label='Predicted')
plt.scatter(y_test[:, 0], y_test[:, 0], color='g', label='Actual', alpha=0.5)
plt.title('Actual vs Predicted - X')
plt.xlabel('Actual X')
plt.ylabel('Predicted X')
plt.legend()
plt.grid()

# Plot actual vs predicted for y
plt.subplot(3, 1, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1], color='r', label='Predicted')
plt.scatter(y_test[:, 1], y_test[:, 1], color='g', label='Actual', alpha=0.5)
plt.title('Actual vs Predicted - Y')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.legend()
plt.grid()

# Plot actual vs predicted for z
plt.subplot(3, 1, 3)
plt.scatter(y_test[:, 2], y_pred[:, 2], color='r', label='Predicted')
plt.scatter(y_test[:, 2], y_test[:, 2], color='g', label='Actual', alpha=0.5)
plt.title('Actual vs Predicted - Z')
plt.xlabel('Actual Z')
plt.ylabel('Predicted Z')
plt.legend()
plt.grid()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()




# Predicted values for the test set
y_pred = X_test @ theta

# Compute the error for each test example (difference between predicted and actual values)
error = y_pred - y_test

group_size = 20

# Number of groups
num_groups = len(error) // group_size

# Initialize arrays to store means and standard deviations
mean_error_x = []
mean_error_y = []
mean_error_z = []
std_error_x = []
std_error_y = []
std_error_z = []

# Loop through each group and calculate mean and standard deviation
for i in range(num_groups):
    start = i * group_size
    end = start + group_size

    # Get the current group of errors
    group_error = error[start:end, :]
    
    # Compute mean for each dimension (x, y, z)
    mean_error_x.append(np.mean(group_error[:, 0]))
    mean_error_y.append(np.mean(group_error[:, 1]))
    mean_error_z.append(np.mean(group_error[:, 2]))

    # Compute standard deviation for each dimension (x, y, z)
    std_error_x.append(np.std(group_error[:, 0]))
    std_error_y.append(np.std(group_error[:, 1]))
    std_error_z.append(np.std(group_error[:, 2]))

# Convert lists to arrays for plotting
mean_error_x = np.array(mean_error_x)
mean_error_y = np.array(mean_error_y)
mean_error_z = np.array(mean_error_z)
std_error_x = np.array(std_error_x)
std_error_y = np.array(std_error_y)
std_error_z = np.array(std_error_z)

# Create subplots
fig, ax = plt.subplots(3, 1, figsize=(8, 7))

# Plot mean and standard deviation for x
ax[0].errorbar(range(num_groups), mean_error_x, yerr=std_error_x, fmt='r-', label='Mean error for x with std dev')
ax[0].set_title('Grouped Mean and Standard Deviation of Error - X')
ax[0].set_xlabel('Group Index (20 examples per group)')
ax[0].set_ylabel('Error in X')
ax[0].grid(True)
ax[0].legend()

# Plot mean and standard deviation for y
ax[1].errorbar(range(num_groups), mean_error_y, yerr=std_error_y, fmt='g-', label='Mean error for y with std dev')
ax[1].set_title('Grouped Mean and Standard Deviation of Error - Y')
ax[1].set_xlabel('Group Index (20 examples per group)')
ax[1].set_ylabel('Error in Y')
ax[1].grid(True)
ax[1].legend()

# Plot mean and standard deviation for z
ax[2].errorbar(range(num_groups), mean_error_z, yerr=std_error_z, fmt='b-', label='Mean error for z with std dev')
ax[2].set_title('Grouped Mean and Standard Deviation of Error - Z')
ax[2].set_xlabel('Group Index (20 examples per group)')
ax[2].set_ylabel('Error in Z')
ax[2].grid(True)
ax[2].legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(y_pred[1:10, :])
print(y_test[1:10, :])