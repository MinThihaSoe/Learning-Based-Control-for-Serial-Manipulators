import os
import csv
import time
import ffmpeg
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from heartshapetrajectory import HeartTrajectory
from sympy import *
t1, t2, t3, t4, t5, t6, t7, t_i = symbols("\\theta_1 \\theta_2 \\theta_3 \\theta_4 \\theta_5 \\theta_6 \\theta7 \\theta_i")

np.set_printoptions(precision=2, suppress=True)  # Set decimal precision to 2 and suppress scientific notation

class RoboticArm:
    def __init__(self, DHparameters,Joint_Limits):
        # DHparameters is a 4xN matrix where each row represents a specific parameter:
        # [Link_twists, link_offset, link_lengths, Joint_offset]
        self.DHparameters = np.array(DHparameters)

        # Assign Joint_Limits
        self.Joint_Limits = Joint_Limits

    def transformation_matrix(self, alpha_i1, a_i1, d_i, t_i):

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

    def ForwardKinematic(self,Joint_angles):
        # Initialize the total transformation matrix as an identity matrix
        T_total = np.eye(4)
        
        # Compute the transformation matrices for each joint
        for i in range(len(self.DHparameters)):
            alpha_i1 = self.DHparameters[i, 0]
            a_i1 = self.DHparameters[i, 1]
            d_i = self.DHparameters[i, 2]
            t_i = self.DHparameters[i, 3]
            # Calculate the transformation matrix for the current joint
            T_i = self.transformation_matrix(alpha_i1, a_i1, d_i, t_i+Joint_angles[i])

            # Multiply with the total transformation matrix
            T_total = T_total @ T_i
        return T_total[:3, 3]
    
    def compute_jacobian(self, Current_Joint_angles, delta=1e-6):
        """
        Compute the Jacobian matrix using numerical differentiation (finite differences).
        
        :param Current_Joint_angles: Array of joint angles [theta1, theta2, ...].
        :param delta: Small perturbation for numerical differentiation.
        :return: Jacobian matrix (6 x n).
        """
        n = len(Current_Joint_angles)  # Number of joints
        Jacobian = np.zeros((6, n))    # Jacobian matrix (6 rows: 3 linear, 3 angular)
        
        # Base origin and Z-axes for Jacobian
        positions = []
        z_vectors = []
        T = np.eye(4)  # Identity matrix for base frame
        
        # Compute forward kinematics to find all transformations
        for i, params in enumerate(self.DHparameters):
            alpha_val, a_val, d_val, dhtheta = params
            Joint_theta = Current_Joint_angles[i]
            Ti = self.transformation_matrix(alpha_val, a_val, d_val, dhtheta + Joint_theta)
            T = np.dot(T, Ti)  # Update transformation
            positions.append(T[:3, 3])  # Extract position
            z_vectors.append(T[:3, 2])  # Extract Z-axis
        
        P_0E = positions[-1]  # End-effector position

        # Numerical differentiation for each joint
        for i in range(n):
            # Perturb current joint angle
            angles_plus = Current_Joint_angles.copy()
            angles_plus[i] += delta
            
            # Compute forward kinematics with perturbed angles
            T_plus = np.eye(4)
            positions_plus = []
            
            for j, params in enumerate(self.DHparameters):
                alpha_val, a_val, d_val, dhtheta = params
                Joint_theta = angles_plus[j]
                Ti = self.transformation_matrix(alpha_val, a_val, d_val, dhtheta + Joint_theta)
                T_plus = np.dot(T_plus, Ti)
                positions_plus.append(T_plus[:3, 3])  # Extract position
            
            P_0E_plus = positions_plus[-1]  # End-effector position with perturbed angles
            
            # Approximate linear velocity (position derivative)
            Jacobian[:3, i] = (P_0E_plus - P_0E) / delta
            
            # Angular velocity is Z-axis of rotation (no finite differences needed)
            Jacobian[3:, i] = z_vectors[i]

        return Jacobian

    def InverseKinematic(self, initial_joint_angles, desired_position, max_iterations, Joint_Limits, tolerance=1e-3, learning_rate=0.01, delta=0.01):
        """
        Computes the inverse kinematics using the Jacobian.

        Parameters:
        - desired_position: np.array (3,), target position for the end-effector
        - initial_joint_angles: np.array (n,), initial guess for the joint angles
        - max_iterations: int, maximum number of iterations
        - tolerance: float, error tolerance for stopping
        - learning_rate: float, gain or damping factor
        - Joint_Limits: np.array (n, 2), joint limits for each joint (upper, lower)

        Returns:
        - joint_angles: np.array (n,), computed joint angles
        """
        # Initialize joint angles
        joint_angles = np.array(initial_joint_angles)

        for iteration in range(max_iterations):
            # Step 1: Compute forward kinematics
            current_position = self.ForwardKinematic(joint_angles)

            # Step 2: Compute error
            delta_x = desired_position - current_position

            # Step 3: Check for convergence and angle limits:
            if np.linalg.norm(delta_x) < tolerance:
                # Solution converged
                # print(f"Converged in {iteration + 1} iterations")
                break

            # Step 4: Compute the Jacobian
            J = self.compute_jacobian(joint_angles)
            # Use only the linear velocity part of the Jacobian (top 3 rows)
            J_pos = J[:3, :]
            
            # Step 5: Compute joint angle update based on the pseudoinverse method
            J_pseudo = np.linalg.pinv(J_pos)  # Compute the pseudoinverse
            delta_theta = learning_rate * J_pseudo @ delta_x

            # Step 6: Update joint angles
            joint_angles = joint_angles + delta_theta

        else:
            print("Maximum iterations reached without convergence.")
        
        return joint_angles
    
    def Compute_jointpositions(self, Joint_angles):

        Link_twists = self.DHparameters[:, 0]
        link_offset = self.DHparameters[:, 1]
        link_lengths = self.DHparameters[:, 2]
        theta_offset = self.DHparameters[:, 3]

        # Compute joint positions
        T = np.eye(4)  # Start with identity matrix 4X4

        positions = [[0, 0, 0]] # Base position
        
        for i, angle in enumerate(Joint_angles):
            T = T @ self.transformation_matrix(Link_twists[i], link_offset[i], link_lengths[i], theta_offset[i] + angle)
            positions = np.vstack((positions, T[:3, 3]))  # Append joint position
        
        # Convert symbolic results to numerical for plotting
        positions = np.round(np.array([[float(coord) for coord in pos] for pos in positions]),2)
        
        return positions  # Return positions as a numpy array
    
    def plot(self, Joint_angles, Target):
        """
        Visualize the robotic arm in 3D space.
        :param theta: Joint angles (in radians).
        :param goal: Desired end-effector position.
        """
        # Plot
        positions = self.Compute_jointpositions(Joint_angles)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', label="Robotic Arm")
        ax.scatter(*Target, color='red', label="End Effector Position", s=10)

        ax.set(xlabel='X', ylabel='Y', zlabel='Z', title='Robotic Arm Visualization')
        ax.legend()
        ax.grid(True)
        plt.show()

    def realtime_visualize(self, initial_angles, trajectory, max_steps=100, tolerance=1e-3, loop=False, loop_count=None):
        print('Starting the animation ...')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize plot elements
        line, = ax.plot([], [], [], '-o', label='Robot Arm', color='black')
        Target = ax.scatter([], [], [], color='Purple', label='Target', s=10, alpha=0.5)
        Trajectory_path, = ax.plot([], [], [], color='red', linestyle='-', linewidth=2, label='Trajectory path')

        # Set workspace limits
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(0, 1500)
        ax.legend()

        # Initialize kinematics and trajectory
        joint_angles = initial_angles.copy()  # Set initial joint angles
        goal = trajectory.get_next_point()  # Get the first goal point

        # Initialize robot positions
        initial_positions = self.Compute_jointpositions(joint_angles)  # Get initial positions
        line.set_data(initial_positions[:, 0], initial_positions[:, 1])  # Set initial 2D plot data
        line.set_3d_properties(initial_positions[:, 2])  # Set initial 3D plot data
        Target._offsets3d = ([goal[0]], [goal[1]], [goal[2]])  # Set initial goal marker

        # List to store the previous goal positions
        goal_positions = [goal]  # Start with the initial goal position

        # Loop control variables
        loops_completed = 0
        loop_in_progress = True  # Keeps track of whether the loop should continue

        # Time tracking variables
        last_time = time.time()

        def update(frame):
            nonlocal joint_angles, goal, loops_completed, loop_in_progress, last_time

            current_time = time.time()
            elapsed_time = current_time - last_time  # Time since the last frame
            last_time = current_time  # Update last_time for next frame

            # Dynamically adjust FPS based on elapsed time
            target_interval = 1.0  # Target interval (1 second, can be adjusted)
            remaining_time = target_interval - elapsed_time

            if remaining_time > 0:
                time.sleep(remaining_time)  # Sleep for the remaining time to match target interval

            if loop_in_progress:
                # Forward kinematics: compute current position
                current_pos = self.ForwardKinematic(joint_angles)
                
                self.createcsv('Forwardkin.csv', ['q1', 'q2', 'q3', 'q4', 'q5', 'q6','x', 'y', 'z'])
                rounded_pos = np.round(current_pos, 4)  # Round off the current position to 3 decimal places
                rounded_angles = np.round(joint_angles, 4)  # Round off the joint angles to 3 decimal places
                self.csvsave('Forwardkin.csv',*rounded_angles,*rounded_pos)

                print(f"frame:{frame}/{max_steps}, Current End Effector's position: {current_pos}, Current Joint Angles {joint_angles}")

                # Check if close to goal
                if np.linalg.norm(np.array(current_pos) - np.array(goal)) < tolerance:
                    try:
                        goal = trajectory.get_next_point()
                    except StopIteration:
                        if loop:
                            print(f"Looping back to start after {loops_completed + 1} loops.")
                            trajectory.reset()
                            loops_completed += 1
                        elif loop_count is not None and loops_completed < loop_count:
                            loops_completed += 1
                            if loops_completed < loop_count:
                                trajectory.reset()
                        else:
                            loop_in_progress = False

                # Compute joint angles for the next step using inverse kinematics
                joint_angles = self.InverseKinematic(
                    joint_angles,
                    desired_position=goal,
                    max_iterations=100000,
                    Joint_Limits=self.Joint_Limits,
                    tolerance=tolerance,
                    learning_rate=0.01
                )

                # Update robot arm positions based on the new joint angles
                positions = self.Compute_jointpositions(joint_angles)
                line.set_data(positions[:, 0], positions[:, 1])
                line.set_3d_properties(positions[:, 2])
                Target._offsets3d = ([goal[0]], [goal[1]], [goal[2]])

                # Append the current goal position for the trajectory path
                goal_positions.append(goal)
                goal_path = np.array(goal_positions)
                Trajectory_path.set_data(goal_path[:, 0], goal_path[:, 1])
                Trajectory_path.set_3d_properties(goal_path[:, 2])

            return line, Target, Trajectory_path

        # Create the animation object with dynamic timing
        ani = FuncAnimation(fig, update, frames=max_steps, blit=False)

        # Save animation to a file
        #ani.save('robotic_arm_animation.gif', writer='pillow', fps=1, dpi=600)
        #ani.save('robotic_arm_animation.mp4', writer='ffmpeg', fps=60, dpi=600)
        plt.show()

        if not loop_in_progress:
            print(f"Completed {loops_completed} loops. Ending animation.")

    def createcsv(self, filename, headers):
        try:
            if not os.path.exists(filename):
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                print(f"File '{filename}' created with headers: {headers}")
        except Exception as e:
            print(f"Error creating file '{filename}': {e}")

    def csvsave(self, filename, *values):
        try:
            if os.path.exists(filename):
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(values)
            else:
                print(f"Error: File '{filename}' does not exist. Use 'createcsv' to create it first.")
        except Exception as e:
            print(f"Error saving data to file '{filename}': {e}")

if __name__ == "__main__":

    # IRB 2400 DH parameters [alpha, a, d, theta]
    DHparams = [
        [       0,   0, 615, 0       ],  # Joint 1
        [-np.pi/2, 100,   0, -np.pi/2],  # Joint 2
        [       0, 705,   0, 0       ],  # Joint 3
        [-np.pi/2, 135, 755, 0       ],  # Joint 4
        [ np.pi/2,   0,   0, 0       ],  # Joint 5
        [-np.pi/2,   0,  85, 0       ]   # Joint 6
    ]

    Joint_Limits = np.array([
        [180, -180],
        [100, -110],
        [ 60,  -65],
        [200, -200],
        [120, -120],
        [400, -400]
    ]) * (np.pi / 180)

    # Example Starting Angle and Target End Effector Positon
    Starting_angles = [0, 0, 0, 0, 0, 0]

    # Instantiate robotic arm and trajectory
    IRB2400 = RoboticArm(DHparams, Joint_Limits)
    Heart = HeartTrajectory()

    # Real-time visualize with looping
    IRB2400.realtime_visualize(Starting_angles, trajectory=Heart, max_steps=100, loop=False, loop_count=1)
    
    # Target_Positon = [500, 0, 600] # Compute the Inverse kinematics and plot it
    # final_joint_angles = IRB2400.InverseKinematic(Starting_angles, Target_Positon, 20000, Joint_Limits)
    # IRB2400.plot(final_joint_angles, Target_Positon)
    # IRB2400.compute_jacobian(Starting_angles)
    # for _ in range(5):  # Print the first 5 points # Get the next point in the trajectory
    #      print(Heart.get_next_point())


#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
