function [ simout ] = QuarterModelSimulation(psi, y, sim_time)
% QuarterModelSimulation runs a Simulink model based on provided PSI
% and outputs the relevant data to be used elsewhere

global m_s m_u c_s k_s k_u g alpha zeta

%% Constants
N_over_lb = 4.448;      % [N / lb]
m_over_in = .0254;      % [m / in]
m_over_mm = .001;       % [m /mm]
Pa_over_psi = 6894.76;  % [Pa / psi]
g = 9.81;               % gravity, m/s^2

%% Vehicle parameters (user-provided)
m_s_full = 1109;                    % full body mass, kg
zeta = .25;                         % dampening ratio
epsilon = 8;                        % sprung/unsprung mass ratio
alpha = .1;                         % natural frequency ratio

%% Vehicle parameters (calculated)
m_s = m_s_full / 4;                 % quarter body mass, kg
m_air = CalculateTireWeight(psi);   % mass of air in tire, kg
m_u = (m_s / epsilon) + m_air;      % quarter unsprung mass, kg

%% Calculate suspension values from ideal conditions (32 psi)
[ k_s, c_s, omega_s ] = CalculateSuspensionStiffnessDamping(32);

%% Calculate tire stiffness from PSI
% Unsprung mass refers to all masses that are attached to and not supported by the spring, such as wheel, axle, or brakes.
[ k_u, omega_u ] = CalculateTireStiffness(psi);

%% Check we have all the values we need for the simulation
debug = 0;
if debug
    fprintf('psi = %f [psi]\n', psi);
    fprintf('step_size = %f [m]\n', y);
    fprintf('m_s = %f [kg]\n', m_s);
    fprintf('m_u = %f [kg]\n', m_u);
    fprintf('c_s = %f [N/(m/s)]\n', c_s);
    fprintf('k_s = %f [N/m]\n', k_s);
    fprintf('k_u = %f [N/m]\n', k_u);
    fprintf('g = %f [m/s^2]\n', g);
    % And print out the stuff that we don't need anyways
    fprintf('omega_s = %f [Hz]\n', omega_s);
    fprintf('omega_u = %f [Hz]\n', omega_u);
end

%% run Simulink simulation
sim('QuarterModelMatrix.slx', sim_time);

end