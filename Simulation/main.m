%% Quarter Model Simulation MAIN
% Alex Kost
% Thesis
%
% Main file for quarter model simulation procedure.
%
% Arguments (see 'Test Parameters' section):
%   num_sims = number of simulations to run
%   psi_min = minimum PSI to simulate
%   psi_max = maximum PSI to simulate
%   save_path = path to save the simulation data
%
% Simulation data will output as CSVs and be organized by column:
%   [ time sprung_acc unsprung_acc sprung_height ]
%

%% Reset workspace
clc
clear all
close all

%% Test parameters (user-provided)
num_sims = 21;              % number of simulations to run
psi_min = 20;               % minimum psi to simulate
psi_max = 40;               % maximum psi to simulate
save_path = '/Users/alexkost/Dropbox/Grad Life/thesis/Data/simulation';

%% Test parameters (predefined)
% Create a range of PSIs using defined values above
psi_all = linspace(psi_min, psi_max, num_sims+1);

% ICs for simulations (cannot be nested in functions)
% Should consider making this dynamic... but don't know how
IC = [-1.74412834455962e-12
    -2.44861738501480e-06
    -5.70054231468026e-11
    -7.99748963336152e-05];

%% Run simulations
figure(1)
hold on;

for i=1:num_sims
    psi = psi_all(i);
    simout = QuarterModelSimulation(psi, save_path);
    
    % Output plot to verify things are working - they are!
    time = simout(:,6);
    z = simout(:,1);
    str = strcat(num2str(psi_all(i)), ' psi');
    plot(time, z, 'DisplayName', str);
end

hold off;
title('Quarter-Car motion vs. Time');
xlabel('Time (s)');
ylabel('Car height (m)');
legend('show');
