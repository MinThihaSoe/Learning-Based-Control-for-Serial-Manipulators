import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CyclicLR,ReduceLROnPlateau

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, batch_size, lr, activation_fn, config_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.ManipulatorNet6DoF(input_size, hidden_layers, output_size, activation_fn).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)

        # Initialize the parameters
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.activation_fn = activation_fn
        self.config_name = config_name  # The folder name passed from Run_me.py

        # Set the checkpoint and metadata path to the folder passed by Run_me.py
        self.checkpoint_path = os.path.join(self.config_name, 'checkpoint.pth')
        self.metadata_path = os.path.join(self.config_name, 'metadata.json')

        # Create directories if they don't exist
        os.makedirs(self.config_name, exist_ok=True)

        self.scheduler = CyclicLR(self.optimizer, base_lr=1e-15, max_lr=1e-3, step_size_up=5, step_size_down=5, mode='exp_range', gamma=0.9)
        #self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        # self.scheduler = ReduceLROnPlateau(self.optimizer, 
        # mode='min',               # Minimize the monitored metric (e.g., validation loss)
        # factor=0.1,               # Reduce LR by this factor (90%)
        # patience=0,               # Number of epochs with no improvement to wait
        # threshold=1e-4,           # Minimum change to qualify as improvement
        # cooldown=0,               # Cooldown epochs before resuming normal operation
        # )

        print(f"Using {self.device} for training")

    class ManipulatorNet6DoF(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, activation_fn, dropout_prob=0.5):
            super().__init__()
            self.layers = []
            self.dropouts = []  # Add dropout layers
            previous_size = input_size
            self.activation_fn = activation_fn
            
            for hidden_size in hidden_layers:
                self.layers.append(nn.Linear(previous_size, hidden_size))
                self.dropouts.append(nn.Dropout(p=dropout_prob))  # A Dropout layer with specified probability
                previous_size = hidden_size
                
            self.layers = nn.ModuleList(self.layers)
            self.dropouts = nn.ModuleList(self.dropouts)  # dropouts
            self.out = nn.Linear(previous_size, output_size)

            # Initialize weights and biases
            self.init_weights()

        # Initializing weights function with He or Xavier
        def init_weights(self):
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    if isinstance(self.activation_fn, nn.ReLU) or isinstance(self.activation_fn, nn.LeakyReLU):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # He initialization
                    elif isinstance(self.activation_fn, nn.Tanh) or isinstance(self.activation_fn, nn.Sigmoid):
                        nn.init.xavier_normal_(layer.weight)  # Xavier initialization
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero

            # Initialize the output layer
            if isinstance(self.activation_fn, nn.ReLU) or isinstance(self.activation_fn, nn.LeakyReLU):
                nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
            elif isinstance(self.activation_fn, nn.Tanh) or isinstance(self.activation_fn, nn.Sigmoid):
                nn.init.xavier_normal_(self.out.weight)
            nn.init.zeros_(self.out.bias)  # Initialize biases to zero

        def forward(self, x):
            for layer, dropout in zip(self.layers, self.dropouts):
                x = self.activation_fn(layer(x))
                x = dropout(x)  # Apply dropout after the activation function
            return self.out(x)

    def load_data(self, train_data_path, test_data_path):
        # Load training data
        train_data = np.loadtxt(train_data_path, delimiter=',', skiprows=1)
        train_input_data = train_data[:, :-3]
        train_target_data = train_data[:, -3:]

        # Load test data
        test_data = np.loadtxt(test_data_path, delimiter=',', skiprows=1)
        test_input_data = test_data[:, :-3]
        test_target_data = test_data[:, -3:]

        # Convert to tensors
        self.X_train = torch.tensor(train_input_data, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(train_target_data, dtype=torch.float32).to(self.device)

        self.X_test = torch.tensor(test_input_data, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(test_target_data, dtype=torch.float32).to(self.device)

        # Create DataLoader for the training set
        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train),
                                    batch_size=self.batch_size, shuffle=True)
        print(f"Training data loaded from {train_data_path}")
        print(f"Test data loaded from {test_data_path}")

    def save_checkpoint(self, epoch, test_losses):
        folder_path = self.config_name  # Use the config_name as part of the path
        os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists

        checkpoint_file = os.path.join(folder_path, 'checkpoint.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'Epoch': epoch,
            'test_losses': test_losses
        }, checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path) and os.path.exists(self.metadata_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata['Epoch'], metadata['Test losses']
        return 0, []

    def log_training_progress(self, epoch, loss, test_loss):
        folder_path = self.config_name
        os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists

        metadata_file = os.path.join(folder_path, 'metadata.json')

        # Load the existing metadata, if available
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Initialize metadata if not available
            activation_fn_name = None
            if isinstance(self.activation_fn, nn.LeakyReLU):
                activation_fn_name = "LeakyReLU"
            else:
                activation_fn_name = self.activation_fn.__name__

            metadata = {
                "input_size": self.model.layers[0].in_features,
                "output_size": self.model.out.out_features,
                "layers": [layer.in_features for layer in self.model.layers]
                            + [self.model.layers[-1].out_features]
                            + [self.model.out.out_features],
                "activation_function": activation_fn_name,
                "Epoch": 0,
                "Test losses": [],
                "Train Loss": []
            }

        # Update metadata with the current epoch and losses
        metadata["Epoch"] = epoch
        metadata["Train Loss"].append({"Epoch": epoch, "loss": loss})
        metadata["Test losses"].append({"Epoch": epoch, "loss": test_loss})

        # Save the updated metadata to the file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

    def randomize_data(self):
        # Shuffle only the training data
        indices = torch.randperm(len(self.X_train))
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

        # Update the DataLoader with the shuffled training data
        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs, patience=5):
        """
        Train the neural network model with early stopping based on test loss.
        Stops training if the test loss does not improve for 'patience' consecutive epochs.
        """
        start_epoch, test_losses = self.load_checkpoint()
        print(f"Starting from Epoch {start_epoch}")

        loss_values = [entry['loss'] for entry in test_losses]
        best_loss = min(loss_values) if test_losses else float("inf")  # Default to infinity if test_losses is empty
        
        print(f"Best Test Loss: {best_loss}")
        print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.15f}")

        patience_counter = 0
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0
            total_batches = len(self.train_loader)

            # Use tqdm for progress visualization
            with tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    self.optimizer.zero_grad() # Clears (resets) the gradients of all model parameters
                    outputs = self.model(inputs)
                    batch_loss = self.criterion(outputs, targets)
                    batch_loss.backward()

                    self.optimizer.step() # Updates the model parameters using the gradients computed during .backward()
                    pbar.update(1)
                    pbar.set_postfix(loss=batch_loss.item())

                    epoch_loss += batch_loss.item() * inputs.size(0)

            epoch_loss /= len(self.X_train)

            # Evaluate on the test set
            self.model.eval()
            with torch.no_grad():
                test_predictions = self.model(self.X_test)
                test_loss = self.criterion(test_predictions, self.y_test)
                test_losses.append(test_loss.item())

            # log and print the summary    
            self.log_training_progress(epoch + 1, epoch_loss, test_loss.item())
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Loss: {test_loss.item():.4f}")            

            # Save the model if test loss improves
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                patience_counter = 0
                self.save_checkpoint(epoch + 1, test_losses)
                print(f"New best test loss: {best_loss:.4f}. Checkpoint saved.")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs.")

            # Early stopping if patience is exhausted
            if patience_counter >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

            # Step the scheduler
            self.scheduler.step()
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.15f}")            

            # Randomize data after each epoch (optional)
            self.randomize_data()

        print("Training completed.")
        self.save_checkpoint(num_epochs, test_losses)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test)
            return predictions.cpu().numpy(), self.y_test.cpu().numpy()

    def plot_results(self, predictions, actuals, config_name):
        """
        Plot the predicted vs actual positions for X, Y, and Z coordinates, 
        including the specific configuration in the title.
        """
        
        fig, axes = plt.subplots(3, 1, figsize=(18, 5))
        coordinates = ['X', 'Y', 'Z']
        
        for i, ax in enumerate(axes):
            ax.scatter(actuals[:, i], predictions[:, i], label='Predicted', color='orange', alpha=0.7)
            ax.plot(actuals[:, i], actuals[:, i], label='Reference', color='blue', alpha=0.3)
            ax.set_title(f'{coordinates[i]} Position')
            ax.set_xlabel('True Coordinate Value(mm)')
            ax.set_ylabel('Predicted Position (mm)')
            ax.legend(loc='upper left')

        # Modify the title with the configuration name
        fig.suptitle(f'Performance of {config_name} Neural Network', fontsize=16) 
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config_name, f'Performance_of_{config_name}_Neural_Network_Predictions_vs_Actual.png'), dpi=300)
        plt.close()
    
    def plot_test_loss_from_json(self):
        """
        Plot the test loss over epochs from the metadata JSON file.
        """
        # Load metadata from JSON file
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract test losses and epochs (since epoch numbers are implicitly the index)
        test_losses = [entry['loss'] for entry in metadata.get('Test losses', [])]
        epochs = [entry['Epoch'] for entry in metadata.get('Test losses', [])]

        # Plot test loss over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, test_losses, linestyle='-', color='red', label=f'Test Loss {self.config_name}')
        plt.title('Test Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss (Log Scale)')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Ensure the directory exists
        output_dir = self.config_name  # Directory name from the attribute
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

        # Save the plot
        output_path = os.path.join(output_dir, f'Test_Loss_over_Epochs_for_{self.config_name}_NN.png')  # Added file extension
        print(f"Saving plot to: {output_path}")  # Debugging statement
        plt.savefig(output_path, dpi=300)
        plt.close()

def main(args):
    # Input_size = Number of Joints, Hidden_size = Neurons, Output_size = End Effector coordinate (xyz)
    # Batch size = Split the Dataset into small chunks, lr = Learning rate
    
    act_fn = (
    torch.relu if args.activation_fn == 'ReLU' 
    else torch.tanh if args.activation_fn == 'Tanh' 
    else torch.sigmoid if args.activation_fn == 'Sigmoid'
    else nn.LeakyReLU(negative_slope=0.01) if args.activation_fn == 'LeakyReLU'
    else None
    )
    
    hidden_layers = list(map(int, args.hidden_layers.strip('()').split(',')))
    
    # Create the NeuralNetwork object inside main()
    nn_model = NeuralNetwork(input_size=6, hidden_layers=hidden_layers, output_size=3, batch_size=10000, lr=0.001, activation_fn=act_fn, config_name=args.config_name)
    
    # Load the data and train the model
    nn_model.load_data("IRB2400forward.csv", "testdata.csv")
    nn_model.train(num_epochs=10000, patience=5)
    
    # Train and evaluate
    predictions, actuals = nn_model.evaluate()
    print("Evaluation complete. Plotting results...")

    # Plotting results
    nn_model.plot_results(predictions, actuals, args.config_name)
    nn_model.plot_test_loss_from_json()
    print("Plotting completed")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run Neural Network with different configurations.")
    parser.add_argument('--activation_fn', type=str, required=True, help="Activation function (ReLU or Tanh)")
    parser.add_argument('--hidden_layers', type=str, required=True, help="Comma-separated list of hidden layers, e.g. '(100,100,100)'")
    parser.add_argument('--config_name', type=str, required=True, help="Configuration name for saving results")

    args = parser.parse_args()

    # Call main with parsed arguments
    main(args)