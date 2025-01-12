import json
import matplotlib.pyplot as plt
import os

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Paths to the ReLU files
relu_files = [
    "Tanh(10,10,10)/metadata.json",
    "Tanh(100,100,100)/metadata.json",
    "Tanh(1000,1000,1000)/metadata.json",
    "Tanh(1000,1000,1000,1000)/metadata.json",
    "Tanh(1000,1000,1000,1000,1000)/metadata.json",
    "Tanh(1000,1000,1000,1000,1000,1000)/metadata.json"
]

# Load ReLU data for all sets
relu_sets = {}

# for i in range(9):
#     relu_data = load_json_data(relu_files[i])
#     relu_sets[f"ReLU Set {i+1}"] = relu_data["Test losses"]

for file_path in relu_files:
    folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name
    relu_data = load_json_data(file_path)
    relu_sets[folder_name] = relu_data["Test losses"]


# Plot all ReLU data in one plot
def plot_all_relu_sets(relu_data):
    plt.figure(figsize=(12, 8))

    for name, data in relu_data.items():
        epochs = [entry["Epoch"] for entry in data]
        losses = [entry["loss"] for entry in data]
        plt.plot(epochs, losses, label=name)

    plt.title("Comparison of All Tanh Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend(loc='best', fontsize='small')  # Smaller font for better spacing
    plt.grid(True)
    plt.show()

# Call the function to plot
plot_all_relu_sets(relu_sets)
