import numpy as np
import matplotlib.pyplot as plt

class HeartTrajectory:
    def __init__(self, x_start=500, y_start=0, z_start=500, num_points=100, loop=False, loop_count=None, Scale=10):
        """
        Initialize the heart trajectory generator.
        :param x_start: Starting X coordinate of the heart.
        :param y_start: Starting Y coordinate of the heart.
        :param z_start: Starting Z coordinate of the heart.
        :param num_points: Number of points in the heart trajectory.
        :param loop: Boolean flag to determine if the trajectory should loop.
        :param loop_count: Number of times to loop. If None, loop indefinitely.
        """
        self.t = np.linspace(np.pi, 3 * np.pi, num_points)
        self.x_start = x_start
        self.y_start = y_start
        self.z_start = z_start
        self.index = 0
        self.loop = loop
        self.loop_count = loop_count
        self.loops_completed = 0
        
        # Parametric equations for the heart in the YZ plane
        h_y = (-(16 * np.sin(self.t)**3))*Scale
        h_z = (13 * np.cos(self.t) - 5 * np.cos(2 * self.t) - 2 * np.cos(3 * self.t) - np.cos(4 * self.t))*Scale

        # Offset the trajectory to start at the given start point
        self.h_y = h_y - h_y[0] + y_start
        self.h_z = h_z - h_z[0] + z_start

    def get_next_point(self):
        """
        Get the next point in the heart trajectory.
        :return: A tuple (x, y, z) representing the next point.
        """
        if self.index >= len(self.t):
            # If looping is enabled and loop_count is not reached, reset the trajectory
            if self.loop and (self.loop_count is None or self.loops_completed < self.loop_count):
                self.index = 0
                self.loops_completed += 1
            else:
                raise StopIteration("End of trajectory reached.")
        
        # The heart shape lies in the YZ plane, so X remains constant
        point = (self.x_start, self.h_y[self.index], self.h_z[self.index])
        self.index += 1
        return point

    def reset(self):
        """Reset the trajectory to the starting point."""
        self.index = 0
        self.loops_completed = 0

    def plot(self):
        """Plot the entire trajectory."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the heart shape
        ax.plot3D(
            self.x_start + np.zeros_like(self.h_y), self.h_y, self.h_z, 
            'r-', label="Heart Shape", linewidth=2
        )
        ax.set_title("Heart Shape Trajectory in YZ Plane")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.grid(True)
        ax.legend()
        plt.show()
