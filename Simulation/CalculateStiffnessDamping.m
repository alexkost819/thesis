function [ k_s, c_s, k_u, omega_u, omega_s ] = CalculateStiffnessDamping(psi)
% Function to identify stiffness and damping coefficients based on tire psi
% Other important parameters are defined by globals
global m_s m_u alpha zeta

%% Constants
Pa_over_psi = 6894.76;  % [Pa / psi]

%% Calculations
k_u_eng = 30.185*psi + 46.375;              % unsprung stiffness, lb/in
k_u = k_u_eng * Pa_over_psi;                % unsprung stiffness, N/m
omega_u = sqrt(k_u/m_u);                    % unsprung natural freq, Hz
k_s = alpha^2 * m_s * omega_u^2;            % sprung stiffness, N/m
omega_s = sqrt(k_s/m_s);                    % sprung natural freq, Hz
c_s = 2 * zeta * sqrt(k_s * m_s);           % spring damping, N/(m/s)

end