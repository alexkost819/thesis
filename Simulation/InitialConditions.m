%%% This file calculates all initial conditions based off tire pressure and
%%% geometry of the car

% assign global values
global kf kr cf cr x0 Mb P_tf P_tr

% Conversion macros for Imp to SI units
N_over_lb = 4.448;      % (N/lb)
m_over_in = .0254;      % (m/in)

% calculations for stiffness and dampening
kf_eng = 30.185*P_tf + 46.375;  % lb/in
kr_eng = 30.185*P_tr + 46.375;  % lb/in
kf = kf_eng * N_over_lb * (1/m_over_in);    % front suspension stiffness in N/m
kr = kr_eng  * N_over_lb * (1/m_over_in);	% rear suspension stiffness in N/m
cf = 2*zeta*sqrt(kf*Mb);                    % front suspension damping in N/(m/s)
cr = 2*zeta*sqrt(kr*Mb);                    % rear suspension damping in N/(m/s)
x0 = [-4.335788328729104e-018;-1.201224489795918e-001;6.462348535570529e-027;-1.033975765691285e-025]; %initial condition