function [ k_u, omega_u ] = CalculateTireStiffness(psi)
% Function to identify stiffness and damping coefficients based on tire psi
% Other important parameters are defined by globals
global m_u

%% Constants
Pa_over_psi = 6894.76;  % [Pa / psi]

%% Calculations
k_u_eng = 30.185*psi + 46.375;              % unsprung stiffness, lb/in
k_u = k_u_eng * Pa_over_psi;                % unsprung stiffness, N/m
omega_u = sqrt(k_u/m_u);                    % unsprung natural freq, Hz

end