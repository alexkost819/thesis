%% Quarter Model Simulation MAIN
% Alex Kost
% Thesis
%
% Main file for quarter model simulation procedure.
%
% Arguments (see 'Test Parameters' section):
%   num_psis = num of psis to simulate
%   psi_min = minimum PSI to simulate
%   psi_max = maximum PSI to simulate
%   num_steps = num of step sizes to simulate
%   step_min = minimum step size to simulate
%   step_max = maximum step size to simulate
%   sim_tim = how long to run the simulation
%   snr = signal-to-noise ratio per sample, dB
%   save_path = path to save the simulation data
%
% Simulation data will output as plots and CSVs

%% Reset workspace and hide figures
clc
clear all
close all
set(0, 'DefaultFigureVisible', 'off');
set(0, 'DefaultFigureWindowStyle', 'docked');

%% Test parameters (user-provided)
num_psi = 27;               % number of psis to simulate
psi_min = 25.5;               % minimum psi
psi_max = 38.5;               % maximum psi

num_steps = 39;             % number of step sizes to simulate
step_min = .1;              % minimum step size, m
step_max = 2;               % maximum step size, m

sim_time = 1.5;            % simulation time, s
snr = 0;                   % signal-to-noise ratio per sample, dB

% save data path
save_path = '/Users/alexkost/Dropbox/Grad Life/thesis/Data/simulation_rnn/';
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

%% Run simulations and get outputs (CSVs and plots)
for i=1:num_steps
    step_size = steps_all(i);
    figure
    hold on;
    for j=1:num_psi
        % run simulation
        psi = psi_all(j);
        simout = QuarterModelSimulation(psi, ...
                                        step_size, ...
                                        sim_time);

        % Add white gaussian noise if snr > 0
        if snr > 0
            for k=1:size(simout, 2)
                simout(:,k) = awgn(simout(:, k), snr);
            end
        end
                       
        % interpret simulation outputs
        sprung_pos = simout(:,1);
        %sprung_vel = simout(:,2);
        sprung_acc = simout(:,7);
        %unsprung_pos = simout(:,3);
        %unsprung_vel = simout(:,4);
        unsprung_acc = simout(:,8);
        step = simout(:,5);    % constant every run
        time = simout(:,6);    % constant every run

        % Plot individual run
        str = strcat(num2str(psi, '%.1f'), ' psi');
        plot(time, sprung_pos, 'DisplayName', str);
        
        % calculate label value
        if psi < 30
            label_val = 0;
        elseif psi <= 34
            label_val = 1;
        elseif psi > 34
            label_val = 2;
        end

        % Output to CSV
        filename = strcat('Sim_', ...
                          num2str(psi, '%.1f'), 'psi_', ...
                          num2str(step_size, '%.2f'), 'm.csv');
        fullfilename = fullfile(save_path, num2str(label_val), filename);

        % Original format. Not good for tensorflow
        %label = ones(length(simout(:,6)),1) * label_val;
        %M = [time sprung_acc unsprung_acc sprung_height label];
        %csvwrite(fullfilename, M);

        % Modifications done for Tensorflow
        %    use sprung acceleration data only (1 feature)
        %    transpose so each row is independent example
        %    remove first .45 seconds of data
        acc_transposed = [sprung_acc]';
        M = acc_transposed(:,(.45/.001):end);
        label_val_column = ones(size(M, 1),1) * label_val;
        csvwrite(fullfilename, horzcat(label_val_column, M));
    end

    % create figure with step
    plot(time, step,'--','DisplayName','Step');
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