function [ m_air ] = CalculateTireWeight(psi)
% Function to identify tire mass based on psi and
% geometric parameters (coded below)
%
% ...should have understood that the mass of the air is negligible
% http://www.madsci.org/posts/archives/2001-08/998945256.Ch.r.html3

%% Constants
m_over_in = .0254;      % [m / in]
m_over_mm = .001;       % [m /mm]
Pa_over_psi = 6894.76;  % [Pa / psi]
R = 8.314472;           % universal gas constant, Pa-m^3/(mol.K)
% cite: http://www.engineeringtoolbox.com/individual-universal-gas-constant-d_588.html
M = 0.02897;            % molecular weight of air, kg/mol
% cite: http://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
T_c = 25;               % temperature, deg C

%% Tire Parameters: https://tiresize.com/tiresizes/175-65R14.htm
m_rubber = 6.85;                    % mass of rubber, kg
w_mm = 175;                         % section width, mm
ratio = .65;                        % aspect ratio of tire
h_o_eng = 14;                       % diameter, in

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

end