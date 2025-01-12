import json
import matplotlib.pyplot as plt
import os

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Paths to the LeakyReLU files
LeakyReLU_files = [
    "LeakyReLU(10,10,10)/metadata.json",
    "LeakyReLU(100,100,100)/metadata.json",
    "LeakyReLU(1000,1000,1000)/metadata.json",
    "LeakyReLU(1000,1000,1000,1000)/metadata.json",
    "LeakyReLU(1000,1000,1000,1000,1000)/metadata.json",
    "LeakyReLU(1000,1000,1000,1000,1000,1000)/metadata.json"
]

# Load LeakyReLU data for all sets with folder-based labels
LeakyReLU_sets = {}

for file_path in LeakyReLU_files:
    folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name
    LeakyReLU_data = load_json_data(file_path)
    LeakyReLU_sets[folder_name] = LeakyReLU_data["Test losses"]

# Plot all LeakyReLU data in one plot
def plot_all_LeakyReLU_sets(LeakyReLU_data):
    plt.figure(figsize=(12, 8))

    for name, data in LeakyReLU_data.items():
        epochs = [entry["Epoch"] for entry in data]
        losses = [entry["loss"] for entry in data]
        plt.plot(epochs, losses, label=name)  # Use folder name as label

    plt.title("Comparison of All LeakyReLU Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend(loc='best', fontsize='small')  # Smaller font for better spacing
    plt.grid(True)
    plt.show()

# Call the function to plot
plot_all_LeakyReLU_sets(LeakyReLU_sets)