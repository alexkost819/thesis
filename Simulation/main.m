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
num_psi = 21;               % number of psis to simulate
psi_min = 20;               % minimum psi
psi_max = 40;               % maximum psi

num_steps = 10;             % number of step sizes to simlate
step_min = .1;               % minimum step size, m
step_max = 1;               % maximum step size, m

sim_time = 1.75;               % simulation time, s

% path to save data to
save_path = '/Users/alexkost/Dropbox/Grad Life/thesis/Data/simulation';

%% Test parameters (predefined)
% Create a range of PSIs and Steps using defined values above
psi_all = linspace(psi_min, psi_max, num_psi);
steps_all = linspace(step_min, step_max, num_steps);

% ICs for simulations (cannot be nested in functions)
% Should consider making this dynamic... but don't know how
IC = [-1.74412834455962e-12
    -2.44861738501480e-06
    -5.70054231468026e-11
    -7.99748963336152e-05];

%% Run simulations and plot figure
for i=1:num_steps
    step_size = steps_all(i);

    figure(i)
    hold on;

    for j=1:num_psi
        psi = psi_all(j);
        simout = QuarterModelSimulation(psi, ...
                                        step_size, ...
                                        save_path, ...
                                        sim_time);
        
        time = simout(:,6);
        z = simout(:,1);
        str = strcat(num2str(psi_all(j)), ' psi');
        plot(time, z, 'DisplayName', str);
    end
    
    % create figure with step
    plot(time, simout(:,5),'--','DisplayName','Step');
    hold off;
    title(sprintf('Quarter-Car Motion\nStep size = %g [m]', step_size));
    xlabel('Time (s)');
    ylabel('Vehicle height (m)');
    legend('show');
    
    % save figure
    filename = sprintf('Plot_step_size_%g.png', step_size);
    fullfilename = fullfile(save_path, filename);
    print(figure(i),fullfilename,'-dpng','-r300')
end