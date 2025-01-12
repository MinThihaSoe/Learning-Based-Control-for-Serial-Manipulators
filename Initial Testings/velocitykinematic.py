import sympy as sp
from sympy import Matrix, cos, sin, pi

# Define symbolic variables
t1, t2, t3, t4, t5, t6 = sp.symbols('theta_1, theta_2, theta_3, theta_4, theta_5, theta_6')
l1, l2, l3, l4, l5, l6 = sp.symbols('l1, l2, l3, l4, l5, l6')  # Link lengths
print(t1)

# Define DH parameters for the UR3 robot
# dh parameters (theta, d, a, alpha)
# Joint 1 to 6
dh_params = [
    (t1         , 0.183 , 0     , pi/2  ),   # Joint 1
    (t2+(pi/2)  , 0     , 0.73  , 0     ),   # Joint 2
    (t3         , 0     , 0.38  , 0     ),   # Joint 3
    (t4-(pi/2)  , 0.0955, 0     , -pi/2 ),   # Joint 4
    (t5         , 0.1155, 0     , pi/2  ),   # Joint 5
    (t6         , 0.0768, 0     , 0     ),   # Joint 6
]

# Transformation matrices
def transformation_matrix(theta, d, a, alpha):
    return Matrix([
        [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Compute transformation matrices for each joint
T_matrices = []
for params in dh_params:
    T_matrices.append(transformation_matrix(*params))

# Calculate the overall transformation matrix from base to end-effector
T_total = Matrix.eye(4)  # Start with the identity matrix
for T in T_matrices:
    T_total = T_total * T

# Extract the position vector (end-effector position)
position = T_total[0:3, 3]

# Compute the Jacobian
# Position is a 3D vector, so we need the Jacobian to map joint velocities to end-effector velocity
jacobian = Matrix.zeros(6, 6)

# Define the rotation matrices for each joint
z_vectors = [Matrix([0, 0, 1])]  # Initial Z-axis for the base
p_vectors = [Matrix([0, 0, 0])]  # Initial position for the base

for i in range(6):
    if i > 0:
        # Update z_vector and position_vector based on previous transformations
        z_vectors.append(T_matrices[i-1][0:3, 2])  # Rotation axis for joint i
        p_vectors.append(T_matrices[i-1][0:3, 3])  # Position vector of the joint i

# Now calculate the Jacobian matrix (linear velocity and angular velocity)
for i in range(6):
    z = z_vectors[i]  # Rotation axis for joint i
    p = p_vectors[-1] - p_vectors[i]  # Position vector for the end-effector relative to the current joint
    jacobian[0:3, i] = z.cross(p)  # Linear velocity part
    jacobian[3:6, i] = z  # Angular velocity part

# Display the Jacobian
sp.pprint(jacobian, use_unicode=True)
