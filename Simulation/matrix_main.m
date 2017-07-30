%% Quarter Model Suspension Simulation
%%% Alex Kost

%% Reset workspace
clc
clear all
close all

global m_s m_u c_s k_s k_u g

%% Constants
N_over_lb = 4.448;      % [N / lb]
m_over_in = .0254;      % [m / in]
m_over_mm = .001;       % [m /mm]
Pa_over_psi = 6894.76;  % [Pa / psi]
R = 8.314472;           % universal gas constant, Pa-m^3/(mol.K)
% cite: http://www.engineeringtoolbox.com/individual-universal-gas-constant-d_588.html
M = 0.02897;            % molecular weight of air, kg/mol
% cite: http://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
T_c = 25;               % temperature, deg C
g = 9.81;               % gravity, m/s^2

%% Initial parameters (user-provided)
% Vehicle parameters
num_pts = 11;                       % number of pressures to test
psi_all = linspace(20,40,num_pts);  % tire pressure, psi
zeta = .25;                         % dampening ratio
m_s = 1109 / 4;                     % body mass, kg
m_driver = 62 / 4;                  % driver mass, kg, NOT USED
m_u_extra = m_s * .15;              % unsprung mass (suspension, axles, etc.), kg
% Unsprung mass refers to all masses that are attached to and not supported by the spring, such as wheel, axle, or brakes.

% Tire Parameters
% https://tiresize.com/tiresizes/175-65R14.htm
m_rubber = 6.85;                    % mass of rubber, kg
w_mm = 175;                         % section width, mm
ratio = .65;                        % aspect ratio of tire
h_o_eng = 14;                       % diameter, in
psi = 32;                           % tire pressure, psi


%% Calculate tire weight
w = w_mm * m_over_mm;               % width of tire, m
sidewall = w * ratio;               % height of tire, m
h_o = h_o_eng * m_over_in;          % outer diameter, m
h_i = h_o - sidewall;               % inner diameter, m
v_tire = pi() * w * ((h_o^2) - (h_i^2))/4;  % volume in tire, m^3

P = psi * Pa_over_psi;              % pressure of tire, Pa
T = T_c + 273.15;                   % temperature, K
rho_air = (P*M)/(R*T);              % density of air, kg/m^3
m_air = v_tire * rho_air;           % mass of tire air, kg
% I should have understood that the mass of the air is negligible
% http://www.madsci.org/posts/archives/2001-08/998945256.Ch.r.html3

m_u = m_u_extra + m_rubber + m_air;             % total mass of tire, kg

%% Calculate stiffness and damping coefficients
k_u_eng = 30.185*psi + 46.375;              % stiffness, lb/in
k_u = k_u_eng * Pa_over_psi;                % stiffness, N/m
c_s = 2 * zeta * sqrt(k_u*m_s);             % damping in N/(m/s)
k_s = k_u * 9;

%% run Simulink simulation for 10 seconds
IC = [-1.74412834455962e-12,
    -2.44861738501480e-06,
    -5.70054231468026e-11,
    -7.99748963336152e-05];

sim('QuarterModelMatrix.slx', 10);

%% output plot
time = simout(:,6);
z = simout(:,1);

figure(1)
plot(time, z);
title('Simulated z-axis height');
xlabel('Time (s)');
%xlim([0 10]);
ylabel('Z height (m)');

%% Run multiple simulations - Not yet...
% for i=1:num_pts
%     for j=1:num_pts
%         % select value from array
%         P_tf = P_tf_all(i);
%         P_tr = P_tr_all(j);
% 
%         % Calculate the rest of the ICs
%         run('InitialConditions.m');
%         
%         % run Simulink simulation for 10 seconds
%         % sim('SimSuspension.slx', 10);
%     end
% end