import sympy as sp
from sympy import simplify,pi

# Define symbolic variables for joint angles (θ1 to θ6)
theta = sp.symbols('theta_1:7')
d = sp.symbols('d_1:7')  # link offsets (distances along the Z-axis)
a = sp.symbols('a_1:7')  # link lengths (distances along the X-axis)
alpha = sp.symbols('alpha_1:7')  # link twists (angles between Z-axes)

# Define transformation matrices based on the DH parameters
def dh_transform(a, alpha, d, theta):
    """Compute individual transformation matrix using DH parameters."""
    return sp.Matrix([
        [sp.cos(theta)               , -sp.sin(theta)               , 0                             , a],
        [sp.sin(theta)*sp.cos(alpha) , sp.cos(theta)*sp.cos(alpha)  , -sp.sin(alpha)                , -d*sp.sin(alpha)],
        [sp.sin(theta)*sp.sin(alpha) , sp.cos(theta)*sp.sin(alpha)  , sp.cos(alpha)                 , d*sp.cos(alpha)],
        [0                           , 0                            , 0                             , 1]
    ])

# Compute forward kinematics for each joint
T = sp.eye(4)  # Start with identity matrix for the base frame
T_matrices = []  # Store transformation matrices

positions = [sp.Matrix([0, 0, 0])]  # Initial origin of the base frame
z_vectors = []  # Initial Z-axis of the base frame

T = sp.eye(4)  # Identity matrix for the base frame
T_matrices = []  # Store transformation matrices

for i in range(len(theta)):
    Ti = dh_transform(a[i], alpha[i], d[i], theta[i])
    T = T * Ti
    T_matrices.append(T)
    positions.append(T[:3, 3])  # Extract origin of the current frame
    z_vectors.append(T[:3, 2])  # Extract Z-axis of the current frame

# Compute the Jacobian matrix
J = sp.zeros(6, 6)
P_0E = positions[-1]  # Position of the end effector

for i in range(len(theta)):
    # Linear velocity part
    J[:3, i] = sp.diff(P_0E,theta[i])
    # Angular velocity part
    J[3:, i] = z_vectors[i]

# UR3 DH parameters
modified_dh_params = {
    theta[0]: theta[0]          , d[0]: 615     , a[0]: 0       , alpha[0]: 0,
    theta[1]: theta[1] - pi / 2 , d[1]: 0       , a[1]: 100     , alpha[1]: -pi/2,
    theta[2]: theta[2]          , d[2]: 0       , a[2]: 705     , alpha[2]: 0,
    theta[3]: theta[3]          , d[3]: 755     , a[3]: 135     , alpha[3]: -pi/2,
    theta[4]: theta[4]          , d[4]: 0       , a[4]: 0       , alpha[4]: pi/2,
    theta[5]: theta[5]          , d[5]: 85      , a[5]: 0       , alpha[5]: -pi/2
}

# Substitute DH parameters into the Jacobian
J_substituted = J.subs(modified_dh_params)
J_simplified = simplify(J_substituted)

theta_values = {
    theta[0]: 0,
    theta[1]: 0,
    theta[2]: 0,
    theta[3]: 0,
    theta[4]: 0,
    theta[5]: 0
}

J_sub = J_simplified.subs(theta_values)
J_simplified = simplify(J_sub)
sp.pprint(J_simplified)

# Simplify the last transformation matrix with substituted values
T_last = T_matrices[-1].subs(modified_dh_params).subs(theta_values)

# Print the last transformation matrix
print("Last Transformation Matrix (T_6):")
sp.pprint(T_last)