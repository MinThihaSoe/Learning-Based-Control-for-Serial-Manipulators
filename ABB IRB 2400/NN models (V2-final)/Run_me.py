import os
import subprocess
import torch

# Define your configurations example below:
#activation_functions = ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid' ]

activation_functions = ['Sigmoid']
hidden_layer_configurations = [
    (1000, 1000, 1000),
    (1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000),
    (1000, 1000, 1000, 1000, 1000, 1000),
]

def generate_folder_name(activation_fn, hidden_layers):
    """Generate a folder name based on activation function and hidden layer configuration."""
    hidden_layers_str = ','.join(map(str, hidden_layers))  # Join with commas
    return f"{activation_fn}({hidden_layers_str})"  # Format folder name

def run_script(script_name, *args):
    """Run a Python script with the given arguments."""
    command = ['python', script_name] + list(args)
    subprocess.run(command, check=True)

def create_and_run():
    """Run the data generation script once and then execute NN scripts for all configurations."""
    
    if not os.path.isfile('IRB2400forward.csv'):
        print("Checking for CUDA availability...")
        if torch.cuda.is_available():
            print("CUDA is available. Running IRB2400_Datageneration_gpu.py...")
            run_script('IRB2400_Datageneration_gpu.py')
        else:
            print("CUDA is not available. Running IRB2400_Datageneration_cpu.py...")
            run_script('IRB2400_Datageneration_cpu.py')
        print("Data generation completed.")
    else:
        print("'IRB2400forward.csv' already exists. Skipping data generation.")
        
    # Run the neural network scripts for all configurations
    for activation_fn in activation_functions:
        for hidden_layers in hidden_layer_configurations:
            # Generate the folder name
            folder_name = generate_folder_name(activation_fn, hidden_layers)
            
            # Create the directory for the configuration
            os.makedirs(folder_name, exist_ok=True)
            print(f"Created directory: {folder_name}")
            
            # Run the neural network script with the specified configuration
            run_script(
                'main_NN.py',
                '--activation_fn', activation_fn, 
                '--hidden_layers', f"({','.join(map(str, hidden_layers))})", 
                '--config_name', folder_name
            )

if __name__ == "__main__":
    create_and_run()