% SLDEMO_SUSPDAT
% SLDEMO_SUSPGRAPH
% sldemo_suspn

clc
clear all
close all

% assign global values
global P_tf P_tr zeta Lf Lr Mb Iyy

% Initial parameters (user-provided)
num_pts = 11;
P_tf_all = linspace(20,40,num_pts);               % front tire pressure
P_tr_all = linspace(20,40,num_pts);               % rear tire pressure
zeta = .25;              % dampening ratio
Lf = 1.06;               % front hub displacement from body CG
Lr = 1.35;               % rear hub displacement from body CG
Mb = 1109;               % body mass in kg
Iyy = 1667.63;           % body moment of inertia about y-axis in kgm^2

% for now
P_tf = 32;
P_tr = 32;  

% Calculate the rest of the ICs
run('InitialConditions.m');

% run Simulink simulation for 10 seconds
sim('SimSuspension.slx', 10);

% Not yet...
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