function [ xDD ] = SimFunc(u)

global m_s m_u c_s k_s k_u g

% To make code readable, reassign Simulink values
x_s = u(1);         % (m)
x_s_d = u(2);       % (m/s)
x_u = u(3);         % (m)
x_u_d = u(4);       % (m/s)
y = u(5);           % (m)

% Assign matrix elements
M11 = m_s;
M12 = 0;
M21 = 0;
M22 = m_u;

C11 = c_s;
C12 = -c_s;
C21 = -c_s;
C22 = c_s;

K11 = k_s;
K12 = -k_s;
K21 = -k_s;
K22 = k_s + k_u;

F11 = m_u*(-g);
F21 = k_u*y + m_s*(-g);

% Assemble matrices
M = [M11 M12;
     M21 M22];

C = [C11 C12;
     C21 C22];

K = [K11 K12;
     K21 K22];

F = [F11;
     F21];

X_d = [x_s_d;
       x_u_d];

X = [x_s;
     x_u];

% Assemble the matrix form of the equation of motion
A = F - (C*X_d) - (K*X);

% Calculating x_s_ddot and x_u_ddot
% https://www.mathworks.com/help/matlab/ref/mldivide.html
xDD = M\A;

% % Equation form
% F_s = -k_s*(x_s - x_u) - c_s*(x_s_d - x_u_d);
% F_u = k_s*(x_s - x_u) + c_s*(x_s_d - x_u_d) - k_u*(x_u - y);
% 
% xDD = [F_s/m_s;
%      F_u/m_u];
% disp(xDD);

end