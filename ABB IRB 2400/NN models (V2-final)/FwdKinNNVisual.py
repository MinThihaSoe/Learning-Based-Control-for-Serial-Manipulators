import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from matplotlib.animation import FuncAnimation
np.set_printoptions(precision=2, suppress=True)

# Robot Arm Class
class RobotArm:
    def __init__(self, DHparams):
        self.DHparams = np.array(DHparams)

    def transformation_matrix(self, alpha_i1, a_i1, d_i, t_i):
        """
        Compute the transformation matrix for a single joint based on DH parameters.
        """
        # Rotation matrix around x-axis (Rx)
        Rx = np.array([[1, 0, 0, 0],
                    [0, np.cos(alpha_i1), -np.sin(alpha_i1), 0],
                    [0, np.sin(alpha_i1), np.cos(alpha_i1), 0],
                    [0, 0, 0, 1]])

        # Translation matrix along x-axis (Dx)
        Dx = np.array([[1, 0, 0, a_i1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        # Rotation matrix around z-axis (Rz)
        Rz = np.array([[np.cos(t_i), -np.sin(t_i), 0, 0],
                    [np.sin(t_i), np.cos(t_i), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        # Translation matrix along z-axis (Dz)
        Dz = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, d_i],
                    [0, 0, 0, 1]])

        # Compute the final transformation matrix
        T_final = Rx @ Dx @ Rz @ Dz
        return T_final

    def compute_joint_positions(self, joint_angles):
        """
        Compute joint positions based on DH parameters and joint angles.
        """
        Link_twists = self.DHparams[:, 0]
        link_offset = self.DHparams[:, 1]
        link_lengths = self.DHparams[:, 2]
        theta_offset = self.DHparams[:, 3]

        # Compute joint positions
        T = np.eye(4)  # Start with identity matrix 4X4
        positions = [[0, 0, 0]]  # Base position
        
        for i, angle in enumerate(joint_angles):
            T = T @ self.transformation_matrix(Link_twists[i], link_offset[i], link_lengths[i], theta_offset[i] + angle)
            positions = np.vstack((positions, T[:3, 3]))  # Append joint position
        
        # Convert symbolic results to numerical for plotting
        positions = np.round(np.array([[float(coord) for coord in pos] for pos in positions]), 2)
        return positions  # Return positions as a numpy array

    def forward_kinematic(self, joint_angles):
        """
        Calculate the end-effector position using forward kinematics.
        """
        T_total = np.eye(4)  # Initialize the total transformation matrix as an identity matrix
        for i in range(len(self.DHparams)):
            alpha_i1 = self.DHparams[i, 0]
            a_i1 = self.DHparams[i, 1]
            d_i = self.DHparams[i, 2]
            t_i = self.DHparams[i, 3]
            # Calculate the transformation matrix for the current joint
            T_i = self.transformation_matrix(alpha_i1, a_i1, d_i, t_i + joint_angles[i])
            # Multiply with the total transformation matrix
            T_total = T_total @ T_i
        
        return T_total[:3, 3]  # End-effector position (XYZ)

# Visualize Class
class Visualize:
    def __init__(self, robot_arm):
        """
        Initialize the Visualize class.
        :param robot_arm: An instance of the RobotArm class.
        """
        self.robot_arm = robot_arm

    def animate(self, joint_angles_list):
        """
        Animate the robot's movement based on a list of joint angles.
        :param joint_angles_list: List of joint angle configurations (list of lists or 2D numpy array).
        """
        # Initialize the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-1000, 1000])
        ax.set_zlim([0, 1000])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm Animation")

        # Line object to represent the robot's links
        line, = ax.plot([], [], [], 'o-', lw=2)

        # Line object for the trajectory of the end-effector
        trajectory, = ax.plot([], [], [], 'r-', lw=1, label="End-Effector Trajectory")
        end_effector_positions = []  # Store end-effector positions for trajectory

        def init():
            """Initialize the animation."""
            line.set_data([], [])
            line.set_3d_properties([])
            trajectory.set_data([], [])
            trajectory.set_3d_properties([])
            return line, trajectory

        def update(frame):
            """Update the robot configuration and trajectory for each frame."""
            joint_angles = joint_angles_list[frame]
            positions = self.robot_arm.compute_joint_positions(joint_angles)
            end_effector_positions.append(self.robot_arm.forward_kinematic(joint_angles))

            # Update the robot's links
            line.set_data(positions[:, 0], positions[:, 1])
            line.set_3d_properties(positions[:, 2])

            # Update the end-effector trajectory
            trajectory_positions = np.array(end_effector_positions)
            trajectory.set_data(trajectory_positions[:, 0], trajectory_positions[:, 1])
            trajectory.set_3d_properties(trajectory_positions[:, 2])

            return line, trajectory

        # Create the animation
        ani = FuncAnimation(
            fig, update, frames=len(joint_angles_list),
            init_func=init, blit=False, interval=200
        )

        ax.legend()
        plt.show()

    def animate_with_predictions(self, joint_angles_list, nn_predictions, config):
        """
        Animate the robot's movement and NN predictions.
        :param joint_angles_list: List of joint angle configurations (list of lists or 2D numpy array).
        :param nn_predictions: Predicted end-effector positions from the neural network.
        """
        # Initialize the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-1000, 1000])
        ax.set_zlim([0, 1000])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm Animation with NN Predictions")

        # Line object to represent the robot's links
        line, = ax.plot([], [], [], 'o-', lw=2)

        # Line object for the trajectory of the end-effector
        trajectory, = ax.plot([], [], [], 'r-', lw=1, label="True End-Effector Trajectory")
        nn_trajectory, = ax.plot([], [], [], '-', lw=1, label=f"{config} NN Predicted Trajectory", color='purple')
        
        end_effector_positions = []  # Store true end-effector positions for trajectory
        nn_trajectory_positions = []  # Store NN predicted positions for trajectory

        def init():
            """Initialize the animation."""
            line.set_data([], [])
            line.set_3d_properties([])
            trajectory.set_data([], [])
            trajectory.set_3d_properties([])
            nn_trajectory.set_data([], [])
            nn_trajectory.set_3d_properties([])
            return line, trajectory, nn_trajectory

        def update(frame):
            """Update the robot configuration and trajectory for each frame."""
            joint_angles = joint_angles_list[frame]
            positions = self.robot_arm.compute_joint_positions(joint_angles)
            end_effector_positions.append(self.robot_arm.forward_kinematic(joint_angles))

            # Update the robot's links
            line.set_data(positions[:, 0], positions[:, 1])
            line.set_3d_properties(positions[:, 2])

            # Update the true end-effector trajectory
            trajectory_positions = np.array(end_effector_positions)
            trajectory.set_data(trajectory_positions[:, 0], trajectory_positions[:, 1])
            trajectory.set_3d_properties(trajectory_positions[:, 2])

            # Update the NN predicted trajectory
            nn_position = nn_predictions[frame]
            nn_trajectory_positions.append(nn_position)  # Add new NN predicted position to the list
            
            nn_trajectory.set_data(np.array(nn_trajectory_positions)[:, 0], np.array(nn_trajectory_positions)[:, 1])
            nn_trajectory.set_3d_properties(np.array(nn_trajectory_positions)[:, 2])

            #plt.savefig(f'animation_frame_{frame:03d}.png', dpi=300)

            return line, trajectory, nn_trajectory

        # Create the animation
        ani = FuncAnimation(
            fig, update, frames=len(joint_angles_list),
            init_func=init, blit=False, interval=200
        )

        ax.legend()
        ani.save('robotic_arm_animation.gif', writer='pillow', fps=60, dpi=600)
        plt.show()

    @staticmethod
    def readcsv(filepath, column_range=None):
        """
        Reads specific columns from a CSV file based on the column_range.

        Parameters:
        - filepath: str, the path to the CSV file.
        - column_range: tuple, specifies the range of columns to extract (start, end).
                        Columns are 0-indexed and the range is [start, end).

        Returns:
        - numpy array of floats, where each row contains values from the specified column range.
        """
        extracted_data = []

        with open(filepath, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header if your file has one

            for row in reader:
                if column_range:
                    extracted_data.append(row[column_range[0]:column_range[1]])
                else:
                    extracted_data.append(row)  # Extract all columns if no range specified

        # Convert to numpy array of floats
        return np.array(extracted_data, dtype=float)

class EvalNN:
    def __init__(self, model_folder_name):
        """
        Initialize the EvalNN class.
        :param folder_path: Path to the folder containing 'checkpoint.pth' and 'metadata.json'.
        """
        self.folder_path = os.path.join(os.getcwd(), model_folder_name)
        self.checkpoint_path = os.path.join(self.folder_path, "checkpoint.pth")
        self.metadata_path = os.path.join(self.folder_path, "metadata.json")
        
        # Determine the device to use (CUDA if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check the checkpoint file
        if not os.path.exists(self.checkpoint_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Checkpoint and metadata not found at {self.folder_path}")        
        self._load_model()

    class ManipulatorNet6DoF(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, activation_fn):

            super().__init__()
            self.layers = []
            previous_size = input_size
            for hidden_size in hidden_layers:
                self.layers.append(nn.Linear(previous_size, hidden_size))
                previous_size = hidden_size
            self.layers = nn.ModuleList(self.layers)
            self.out = nn.Linear(previous_size, output_size)

            # Select the activation function
            if activation_fn.lower() == "relu":
                self.activation_fn = nn.ReLU()  # Correct ReLU instantiation
            elif activation_fn.lower() == "tanh":
                self.activation_fn = nn.Tanh()
            elif activation_fn.lower() == "leakyrelu":
                self.activation_fn = nn.LeakyReLU(negative_slope=0.01)    
            elif activation_fn.lower() == "sigmoid":
                self.activation_fn = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation function: {activation_fn}")

        def forward(self, x):
            for layer in self.layers:
                x = self.activation_fn(layer(x))
            return self.out(x)

    def _load_model(self):
        # Load the metadata from the files
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        input_size = metadata["input_size"]
        output_size = metadata["output_size"]
        activation_fn = metadata["activation_function"].lower()
        layers = metadata["layers"]
        hidden_layers = layers[1:-1]  # Exclude input_size and output_size

        self.model = self.ManipulatorNet6DoF(input_size, hidden_layers, output_size, activation_fn)
        self.model.to(self.device)  # Move model to the correct device (CPU or CUDA)

        # Load the checkpoint and model parameters
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        # Load model state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set the model to evaluation mode
        self.model.eval()

    def predict(self, joint_angles):
        """
        Predict the end-effector position for a given set of joint angles.
        :param joint_angles: A list or numpy array of joint angles.
        :return: Predicted end-effector positions as a numpy array.
        """
        with torch.no_grad():
            # Convert joint angles to torch tensor
            joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32).to(self.device)
            if len(joint_angles_tensor.shape) == 1:
                joint_angles_tensor = joint_angles_tensor.unsqueeze(0)  # Add batch dimension
            
            predictions = self.model(joint_angles_tensor).cpu().numpy()
        return predictions

# Main Function
def main():
    # Define DH parameters for the robot
    DHparams = np.array([
        [0, 0, 615, 0],
        [-np.pi / 2, 100, 0, -np.pi / 2],
        [0, 705, 0, 0],
        [-np.pi / 2, 135, 755, 0],
        [np.pi / 2, 0, 0, 0],
        [-np.pi / 2, 0, 85, 0]
    ])

    # Create RobotArm object
    IRB2400 = RobotArm(DHparams)

    # Create Visualize object
    visualizer = Visualize(IRB2400)

    # Test data for joint angles
    Testdata_path = "Testdata.csv"
    # Extract joint angles and endeffector position from 
    joint_angles = Visualize.readcsv(Testdata_path, column_range=(0, 6))

    # Define configurations for ReLU, Tanh, LeakyReLU, and Sigmoid
    configurations = [
        # ReLU configurations
        #{"name": "ReLU(10,10,10)", "activation": "ReLU", "layers": (10, 10, 10)},
        #{"name": "ReLU(50,50,50)", "activation": "ReLU", "layers": (50, 50, 50)},
        #{"name": "ReLU(50,50,50,50)", "activation": "ReLU", "layers": (50, 50, 50, 50)},
        #{"name": "ReLU(100,100,100)", "activation": "ReLU", "layers": (100, 100, 100)},
        #{"name": "ReLU(100,100,100,100)", "activation": "ReLU", "layers": (100, 100, 100, 100)},
        #{"name": "ReLU(100,100,100,100,100,100)", "activation": "ReLU", "layers": (100, 100, 100, 100, 100, 100)},
        #{"name": "ReLU(256,256,256)", "activation": "ReLU", "layers": (256, 256, 256)},
        #{"name": "ReLU(256,256,256,256,256,256)", "activation": "ReLU", "layers": (256, 256, 256, 256, 256, 256)},
        #{"name": "ReLU(1000,1000,1000,1000,1000,1000)", "activation": "ReLU", "layers": (1000, 1000, 1000, 1000, 1000, 1000)},

        # Tanh configurations
        #{"name": "Tanh(10,10,10)", "activation": "Tanh", "layers": (10, 10, 10)},
        #{"name": "Tanh(50,50,50)", "activation": "Tanh", "layers": (50, 50, 50)},
        #{"name": "Tanh(50,50,50,50)", "activation": "Tanh", "layers": (50, 50, 50, 50)},
        #{"name": "Tanh(100,100,100)", "activation": "Tanh", "layers": (100, 100, 100)},
        #{"name": "Tanh(100,100,100,100)", "activation": "Tanh", "layers": (100, 100, 100, 100)},
        #{"name": "Tanh(100,100,100,100,100,100)", "activation": "Tanh", "layers": (100, 100, 100, 100, 100, 100)},
        #{"name": "Tanh(256,256,256)", "activation": "Tanh", "layers": (256, 256, 256)},
        #{"name": "Tanh(256,256,256,256,256,256)", "activation": "Tanh", "layers": (256, 256, 256, 256, 256, 256)},
        #{"name": "Tanh(1000,1000,1000,1000,1000,1000)", "activation": "Tanh", "layers": (1000, 1000, 1000, 1000, 1000, 1000)},

        # LeakyReLU configurations
        #{"name": "LeakyReLU(10,10,10)", "activation": "LeakyReLU", "layers": (10, 10, 10)},
        #{"name": "LeakyReLU(100,100,100)", "activation": "LeakyReLU", "layers": (100, 100, 100)},
        {"name": "LeakyReLU(1000,1000,1000)", "activation": "LeakyReLU", "layers": (1000, 1000, 1000)}
        #{"name": "LeakyReLU(100,100,100,100)", "activation": "LeakyReLU", "layers": (1000, 1000, 1000, 1000)},
        #{"name": "LeakyReLU(100,100,100,100,100,100)", "activation": "LeakyReLU", "layers": (100, 100, 100, 100, 100, 100)},
        #{"name": "LeakyReLU(256,256,256)", "activation": "LeakyReLU", "layers": (256, 256, 256)},
        #{"name": "LeakyReLU(256,256,256,256,256,256)", "activation": "LeakyReLU", "layers": (256, 256, 256, 256, 256, 256)},
        #{"name": "LeakyReLU(1000,1000,1000,1000,1000,1000)", "activation": "LeakyReLU", "layers": (1000, 1000, 1000, 1000, 1000, 1000)},

        # Sigmoid configurations
        #{"name": "Sigmoid(10,10,10)", "activation": "Sigmoid", "layers": (10, 10, 10)},
        #{"name": "Sigmoid(50,50,50)", "activation": "Sigmoid", "layers": (50, 50, 50)},
        #{"name": "Sigmoid(50,50,50,50)", "activation": "Sigmoid", "layers": (50, 50, 50, 50)},
        #{"name": "Sigmoid(100,100,100)", "activation": "Sigmoid", "layers": (100, 100, 100)},
        #{"name": "Sigmoid(100,100,100,100)", "activation": "Sigmoid", "layers": (100, 100, 100, 100)},
        #{"name": "Sigmoid(100,100,100,100,100,100)", "activation": "Sigmoid", "layers": (100, 100, 100, 100, 100, 100)},
        #{"name": "Sigmoid(256,256,256)", "activation": "Sigmoid", "layers": (256, 256, 256)},
        #{"name": "Sigmoid(256,256,256,256,256,256)", "activation": "Sigmoid", "layers": (256, 256, 256, 256, 256, 256)},
        #{"name": "Sigmoid(1000,1000,1000,1000,1000,1000)", "activation": "Sigmoid", "layers": (1000, 1000, 1000, 1000, 1000, 1000)},
    ]

    for config in configurations:
        print(f"Animating with configuration: {config['name']}")
        eval_nn = EvalNN(config["name"])  # Initialize the neural network evaluator
        nn_predictions = eval_nn.predict(joint_angles)  # Get predictions
        visualizer.animate_with_predictions(joint_angles, nn_predictions, config["name"])  # Animate

    # Animate the robot arm with ReLu(10,10,10) NN
    # eval_nn = EvalNN("ReLu(10,10,10)")
    # nn_predictions = eval_nn.predict(joint_angles)
    # visualizer.animate_with_predictions(joint_angles,nn_predictions,"ReLu(10,10,10)")

    # Animate the robot arm with ReLu(100,100,100) NN
    #eval_nn = EvalNN("ReLu(100,100,100)")
    #nn_predictions = eval_nn.predict(joint_angles)
    #visualizer.animate_with_predictions(joint_angles,nn_predictions,"ReLu(100,100,100)")

    # Animate the robot arm with ReLu(100,100,100,100,100,100) NN
    # eval_nn = EvalNN("ReLu(100,100,100,100,100,100)")
    # nn_predictions = eval_nn.predict(joint_angles)
    # visualizer.animate_with_predictions(joint_angles,nn_predictions,"ReLu(100,100,100,100,100,100)")

    # Animate the robot arm with ReLu(256,256,256,256,256) NN
    #eval_nn = EvalNN("ReLu(256,256,256,256,256)")
    #nn_predictions = eval_nn.predict(joint_angles)
    #visualizer.animate_with_predictions(joint_angles,nn_predictions,"ReLu(256,256,256,256,256)")

    # Animate the robot arm with Tanh(100,100,100) NN
    # eval_nn = EvalNN("Tanh(100,100,100)")
    # nn_predictions = eval_nn.predict(joint_angles)
    # visualizer.animate_with_predictions(joint_angles,nn_predictions,"Tanh(100,100,100)")
    
    #Animate the robot arm with Tanh(256,256,256,256,256,256) NN
    # eval_nn = EvalNN("Tanh(1000,1000,1000,1000,1000,1000)")
    # nn_predictions = eval_nn.predict(joint_angles)
    # visualizer.animate_with_predictions(joint_angles,nn_predictions,"Tanh(1000,1000,1000,1000,1000,1000)")




    # eval_nn = EvalNN("Tanh(100,100,100)")
    # jointangle=[-3.141592653589793,-1.9198621771937625,-1.1344640137963142,-3.490658503988659,-2.0943951023931953,-6.981317007977318]
    # position=[1278.0092382994003,25.176841281712182,370.39756093649055]
    # nn_predictions = eval_nn.predict(jointangle)
    # print(position-nn_predictions)

if __name__ == "__main__":
    main()
