function [ simout ] = QuarterModelSimulation(psi, y, save_path, snr, sim_time)
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
debug = 1;
if debug
    fprintf('psi = %f [psi]\n', psi);
    fprintf('step_size = %f [m]\n', y);
    fprintf('m_s = %f [kg]\n', m_s);
    fprintf('m_u = %f [kg]\n', m_u);
    fprintf('c_s = %f [N/(m/s)]\n', c_s);
    fprintf('k_s = %f [N/m]\n', k_s);
    fprintf('k_u = %f [N/m]\n', k_u);
    fprintf('g = %f [m/s^2]\n', g);
    fprintf('snr = %f [dB]\n', snr);
    % And print out the stuff that we don't need anyways
    fprintf('omega_s = %f [Hz]\n', omega_s);
    fprintf('omega_u = %f [Hz]\n', omega_u);
end

%% run Simulink simulation
sim('QuarterModelMatrix.slx', sim_time);

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
filename = strcat('Sim_', num2str(psi), 'psi_', num2str(y), 'm.csv');
fullfilename = fullfile(save_path, filename);

% calculate label value
if psi < 30
    label_val = 0;
elseif psi <= 34
    label_val = 1;
elseif psi > 34
    label_val = 2;
end
% Create matrix to output
time = simout(:,6);
sprung_acc = acc(:,1);
unsprung_acc = acc(:,2);
sprung_height = simout(:,1);
label = ones(length(simout(:,6)),1) * label_val;

% Add white gaussian noise
if snr > 0
    sprung_acc = awgn(sprung_acc, snr);
    unsprung_acc = awgn(unsprung_acc, snr);
    sprung_height = awgn(sprung_height, snr);
end

M = [time sprung_acc unsprung_acc sprung_height label];

% Write CSV file
csvwrite(fullfilename, M);

end