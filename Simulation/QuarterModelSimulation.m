function [ simout ] = QuarterModelSimulation(psi, save_path)
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
eta = 8;                            % sprung/unsprung mass ratio
alpha = .1;                         % natural frequency ratio

%% Vehicle parameters (calculated)
m_s = m_s_full / 4;                 % quarter body mass, kg
m_u = m_s / eta;                    % quarter unsprung mass, kg

%% Calculate stiffness and damping coefficients
% Unsprung mass refers to all masses that are attached to and not supported by the spring, such as wheel, axle, or brakes.
[ k_s, c_s, k_u, omega_u, omega_s ] = CalculateStiffnessDamping(psi);

%% Check we have all the values we need for the simulation
debug = 0;
if debug
    fprintf('psi = %f [psi]\n', psi);
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

%% run Simulink simulation for 10 seconds
sim('QuarterModelMatrix.slx', 1.5);

%% Output plot to verify things are working - they are!
% time = simout(:,6);
% z = simout(:,1);
% 
% figure(1);
% plot(time, z);
% title('Quarter Car Model');
% xlabel('Time (s)');
% ylabel('Car height (m)');

%% Output data to .csv file
% Identify name and save location
filename = strcat('Simulation_', num2str(psi), '.csv');
fullfilename = fullfile(save_path, filename);

% Create matrix to output
time = simout(:,6);
sprung_acc = acc(:,1);
unsprung_acc = acc(:,2);
sprung_height = simout(:,1);
M = [time sprung_acc unsprung_acc sprung_height];

% Write CSV file
csvwrite(fullfilename, M);

end