# Make sure to run the script in the same directory

import os
import csv
import numpy as np
from tqdm import tqdm

from sympy import symbols, sin, cos, collect, simplify, expand, factor, pprint
from sympy import *
t1, t2, t3, t4, t5, t6, t7, t_i = symbols("\\theta_1 \\theta_2 \\theta_3 \\theta_4 \\theta_5 \\theta_6 \\theta7 \\theta_i")
from sympy import *
a1, a2, a3, d, d1, d2, d3, d4, d5, d6, alpha_i1, a_i1,d_i = symbols("a_1 a_2 a_3 d d_1 d_2 d_3 d_4 d_5 d_6 \\alpha_{i-1} a_{i-1} d_i")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the file name
csv_file = "IRB2400forward.csv"

# Check if the file exists; if not, create it with headers

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write headers
    writer.writerow(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'x', 'y', 'z'])

def transformation_matrix(alpha_i1, a_i1, d_i, t_i):
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
    
    return Rx * Dx * Rz * Dz

#DH parameters for IIRB 2400
alpha_vals = Matrix([0,-pi/2,0,-pi/2,pi/2,-pi/2,0])
a_vals = Matrix([0,100,705,135,0,0,0])
d_vals = Matrix([615,0,0,755,0,85,0])
theta_vals = Matrix([t1,t2-(pi/2),t3,t4,t5,t6,0])

T1 = transformation_matrix(alpha_vals[0],a_vals[0],d_vals[0], theta_vals[0])
T2 = transformation_matrix(alpha_vals[1],a_vals[1],d_vals[1], theta_vals[1])
T3 = transformation_matrix(alpha_vals[2],a_vals[2],d_vals[2], theta_vals[2])
T4 = transformation_matrix(alpha_vals[3],a_vals[3],d_vals[3], theta_vals[3])
T5 = transformation_matrix(alpha_vals[4],a_vals[4],d_vals[4], theta_vals[4])
T6 = transformation_matrix(alpha_vals[5],a_vals[5],d_vals[5], theta_vals[5])
T7 = transformation_matrix(alpha_vals[6],a_vals[6],d_vals[6], theta_vals[6])

T_final = T1 * T2 * T3 * T4 * T5 * T6 * T7
simplify(T_final[0:3,3])

simplified_x = factor(simplify(T_final[0,3]))
collected_x = collect(simplified_x, [cos(t2+t3), sin(t2+t3), cos(t1), sin(t1)])
simplified_y = factor(simplify(T_final[1,3]))
collected_y = collect(simplified_y, [cos(t2+t3), sin(t2+t3), cos(t1), sin(t1)])
simplified_z = factor(simplify(T_final[2,3]))
collected_z = collect(simplified_z, [cos(t2+t3), sin(t2+t3), cos(t1), sin(t1)])

# Collected components in a list
collected_components = ([[collected_x], [collected_y], [collected_z]])

# Uncomment below to check simplified end-effector positions:
#for component in collected_components:
#    pprint(component)

# Define ranges for joint angles
t1_vals = np.linspace(-np.pi, np.pi, 21)                   # theta1 range
t2_vals = np.linspace(-110*np.pi/180, 100*np.pi/180, 21)   # theta2 range
t3_vals = np.linspace(-65*np.pi/180, 60*np.pi/180, 21)     # theta3 range
t4_vals = np.linspace(-200*np.pi/180,200*np.pi/180, 21)    # theta4 range
t5_vals = np.linspace(-120*np.pi/180,120*np.pi/180, 21)    # theta5 range
t6_vals = np.linspace(-400*np.pi/180,400*np.pi/180, 1)     # theta6 range (last theta do not change position)
t7_vals = np.linspace(-np.pi, np.pi, 1)                    # theta7 range (no joint 7 in IRB2400)

# Initialize lists to store end-effector positions
x_positions = []
y_positions = []
z_positions = []

# Initialize lists for positions and joint angles
positions = []
joint_angles = []

# Initialize counter for iterations
counter = 0

# Compute the total number of points
total_points = len(t1_vals) * len(t2_vals) * len(t3_vals) * len(t4_vals) * len(t5_vals) * len(t6_vals) * len(t7_vals)

# Open the CSV file in append mode
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Create a tqdm progress bar
    with tqdm(total=total_points, desc="Processing Points", unit="point") as pbar:
        # Compute end-effector positions for the grid of angles
        for t1_val in t1_vals:
            for t2_val in t2_vals:
                for t3_val in t3_vals:
                    for t4_val in t4_vals:
                        for t5_val in t5_vals:
                            for t6_val in t6_vals:
                                for t7_val in t7_vals:
                                    # Substitute the joint angles
                                    x = simplified_x.subs({t1: t1_val, t2: t2_val, t3: t3_val, t4: t4_val, t5: t5_val, t6: t6_val})
                                    y = simplified_y.subs({t1: t1_val, t2: t2_val, t3: t3_val, t4: t4_val, t5: t5_val, t6: t6_val})
                                    z = simplified_z.subs({t1: t1_val, t2: t2_val, t3: t3_val, t4: t4_val, t5: t5_val, t6: t6_val})
                                    
                                    # Convert symbolic values to floats
                                    pos = [float(x), float(y), float(z)]
                                    angles = [t1_val, t2_val, t3_val, t4_val, t5_val, t6_val]
                                    
                                    # Write joint angles and positions to CSV
                                    writer.writerow(angles + pos)
                                    
                                    # Update the progress bar
                                    pbar.update(1)