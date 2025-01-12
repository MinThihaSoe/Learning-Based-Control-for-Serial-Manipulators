% Using the rvctools and MATLAB
% First run startup_rvc.m

% Define the robot using modified DH parameters
% Link[theta_i d_i a_{i-1} alpha_{i-1}]
L(1) = Link([0 615 0   0]     , 'modified');       % Link 1
L(2) = Link([0 0   100 -pi/2] , 'modified');       % Link 2
L(3) = Link([0 0   705 0]     , 'modified');       % Link 3
L(4) = Link([0 755 135 -pi/2] , 'modified');       % Link 4
L(5) = Link([0 0   0   pi/2]  , 'modified');       % Link 5
L(6) = Link([0 85  0   -pi/2] , 'modified');       % Link 6

% Joint Offset
L(2).offset = -pi/2;

% Create the serial robot
IRB = SerialLink(L, 'name', 'IRB 2400')

% Define the joint angles (6 elements)
joint_angles = [0 0 0 0 0 0];

% Plot the robot with specified joint angles
IRB.plot(joint_angles);
