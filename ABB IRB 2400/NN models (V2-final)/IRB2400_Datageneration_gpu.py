import os
import csv
import numpy as np
import torch
from sympy import symbols, sin, cos, simplify, lambdify, Matrix, pi
from tqdm import tqdm

# Define symbolic variables
t1, t2, t3, t4, t5, t6 = symbols("\\theta_1 \\theta_2 \\theta_3 \\theta_4 \\theta_5 \\theta_6")
alpha_i1, a_i1, d_i, t_i = symbols("\\alpha_{i-1} a_{i-1} d_i \\theta_i")

# Transformation matrix function
def transformation_matrix(alpha_i1, a_i1, d_i, t_i):
    Rx = Matrix([
        [1, 0, 0, 0],
        [0, cos(alpha_i1), -sin(alpha_i1), 0],
        [0, sin(alpha_i1), cos(alpha_i1), 0],
        [0, 0, 0, 1]
    ])
    Dx = Matrix([
        [1, 0, 0, a_i1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Rz = Matrix([
        [cos(t_i), -sin(t_i), 0, 0],
        [sin(t_i), cos(t_i), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Dz = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d_i],
        [0, 0, 0, 1]
    ])
    return Rx * Dx * Rz * Dz

# Define DH parameters for IRB 2400
alpha_vals = Matrix([0, -pi/2, 0, -pi/2, pi/2, -pi/2])
a_vals = Matrix([0, 100, 705, 135, 0, 0])
d_vals = Matrix([615, 0, 0, 755, 0, 85])
theta_vals = Matrix([t1, t2 - pi/2, t3, t4, t5, t6])

# Compute the transformation matrices
T1 = transformation_matrix(alpha_vals[0], a_vals[0], d_vals[0], theta_vals[0])
T2 = transformation_matrix(alpha_vals[1], a_vals[1], d_vals[1], theta_vals[1])
T3 = transformation_matrix(alpha_vals[2], a_vals[2], d_vals[2], theta_vals[2])
T4 = transformation_matrix(alpha_vals[3], a_vals[3], d_vals[3], theta_vals[3])
T5 = transformation_matrix(alpha_vals[4], a_vals[4], d_vals[4], theta_vals[4])
T6 = transformation_matrix(alpha_vals[5], a_vals[5], d_vals[5], theta_vals[5])

# Final transformation matrix
T_final = T1 * T2 * T3 * T4 * T5 * T6
x_expr = simplify(T_final[0, 3])
y_expr = simplify(T_final[1, 3])
z_expr = simplify(T_final[2, 3])

# Export expressions to numerical functions
f_x = lambdify([t1, t2, t3, t4, t5, t6], x_expr, 'numpy')
f_y = lambdify([t1, t2, t3, t4, t5, t6], y_expr, 'numpy')
f_z = lambdify([t1, t2, t3, t4, t5, t6], z_expr, 'numpy')

# Create ranges for joint angles
t1_vals = torch.linspace(-np.pi, np.pi, 21, device='cuda')
t2_vals = torch.linspace(-110*np.pi/180, 100*np.pi/180, 21, device='cuda')
t3_vals = torch.linspace(-65*np.pi/180, 60*np.pi/180, 21, device='cuda')
t4_vals = torch.linspace(-200*np.pi/180,200*np.pi/180, 21, device='cuda')
t5_vals = torch.linspace(-120*np.pi/180,120*np.pi/180, 21, device='cuda')
t6_vals = torch.linspace(-400*np.pi/180,400*np.pi/180, 3, device='cuda')

# Create a grid of joint angles
t1_grid, t2_grid, t3_grid, t4_grid, t5_grid, t6_grid = torch.meshgrid(
    t1_vals, t2_vals, t3_vals, t4_vals, t5_vals, t6_vals, indexing='ij'
)

# Flatten the grid for batch processing
inputs = torch.stack([t1_grid, t2_grid, t3_grid, t4_grid, t5_grid, t6_grid], dim=-1).reshape(-1, 6)

# Define a function to evaluate positions in batches
def evaluate_positions(batch):
    t1, t2, t3, t4, t5, t6 = batch.T
    x = torch.tensor(f_x(t1.cpu().numpy(), t2.cpu().numpy(), t3.cpu().numpy(), t4.cpu().numpy(), t5.cpu().numpy(), t6.cpu().numpy()), device='cuda')
    y = torch.tensor(f_y(t1.cpu().numpy(), t2.cpu().numpy(), t3.cpu().numpy(), t4.cpu().numpy(), t5.cpu().numpy(), t6.cpu().numpy()), device='cuda')
    z = torch.tensor(f_z(t1.cpu().numpy(), t2.cpu().numpy(), t3.cpu().numpy(), t4.cpu().numpy(), t5.cpu().numpy(), t6.cpu().numpy()), device='cuda')
    return torch.cat([batch, torch.stack([x, y, z], dim=1)], dim=1)

# Batch processing for positions
results = []
batch_size = 1024
with tqdm(total=inputs.size(0), desc="Processing Points", unit="point") as pbar:
    for i in range(0, inputs.size(0), batch_size):
        batch = inputs[i:i + batch_size]
        res_batch = evaluate_positions(batch)
        results.append(res_batch)
        pbar.update(batch.size(0))

# Concatenate all results
results = torch.cat(results, dim=0)

# Save joint angles and positions to a CSV file
csv_file = "IRB2400forward.csv"
header = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'x', 'y', 'z']
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(results.cpu().numpy())

print(f"Joint angles and positions saved to {csv_file}")
