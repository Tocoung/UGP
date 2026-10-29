% Set the step size and time span
h = 0.05; % Increased step size to clearly visualize energy drift
t_end = 200;
t_vec = 0:h:t_end;
N_steps = length(t_vec);

% Vectorized anonymous functions
f = @(t,y) [y(2); -sin(y(1))]; 

% The Hamiltonian for the normalized mathematical pendulum
calc_H = @(q, p) 0.5 .* p.^2 - cos(q); 

% ---------------------------------------------------------
% 1. Define a Single Initial Point
% ---------------------------------------------------------
q_init = 0.0; % Initial position (angle)
p_init = 1.0; % Initial momentum

% Set Global Reference Energy for this exact point
H_true = calc_H(q_init, p_init);

% State variables for standard methods
q_euler = q_init; p_euler = p_init;
y_heun  = [q_init; p_init];
y_rk4   = [q_init; p_init]; 
q_sv    = q_init; p_sv    = p_init;
q_sym   = q_init; p_sym   = p_init;

% Initialize arrays to store the Hamiltonian at each time step
H_euler = zeros(1, N_steps);
H_heun  = zeros(1, N_steps);
H_rk4   = zeros(1, N_steps);
H_sv    = zeros(1, N_steps);
H_sym   = zeros(1, N_steps);
H_hr    = zeros(1, N_steps); % Heun with Relaxation

% Calculate and store initial Hamiltonian
H_euler(1) = H_true;
H_heun(1)  = H_true;
H_rk4(1)   = H_true;
H_sv(1)    = H_true;
H_sym(1)   = H_true;
H_hr(1)    = H_true;

% ---------------------------------------------------------
% 2. Integrate and Calculate Hamiltonian (Standard Methods)
% ---------------------------------------------------------
for i = 1:N_steps-1
    ti = t_vec(i);
    
    % --- Explicit Euler Update ---
    p_next_euler = p_euler - h * sin(q_euler);
    q_next_euler = q_euler + h * p_euler; 
    q_euler = q_next_euler;
    p_euler = p_next_euler;
    H_euler(i+1) = calc_H(q_euler, p_euler);
    
    % --- Standard Heun Update ---
    k1_heun = f(ti, y_heun);
    k2_heun = f(ti + h, y_heun + h .* k1_heun);
    y_heun = y_heun + h .* (k1_heun + k2_heun) / 2;
    H_heun(i+1) = calc_H(y_heun(1), y_heun(2));
    
    % --- Standard RK4 Update ---
    k1 = f(ti, y_rk4);
    k2 = f(ti + h/2, y_rk4 + h * k1 / 2);
    k3 = f(ti + h/2, y_rk4 + h * k2 / 2);
    k4 = f(ti + h, y_rk4 + h * k3);
    y_rk4 = y_rk4 + h * (k1 + 2*k2 + 2*k3 + k4) / 6;
    H_rk4(i+1) = calc_H(y_rk4(1), y_rk4(2));
    
    % --- Stormer-Verlet Update ---
    p_half_sv = p_sv - (h / 2) * sin(q_sv);
    q_next_sv = q_sv + h * p_half_sv;
    p_next_sv = p_half_sv - (h / 2) * sin(q_next_sv);
    q_sv = q_next_sv;
    p_sv = p_next_sv;
    H_sv(i+1) = calc_H(q_sv, p_sv);
    
    % --- Symplectic Euler Update ---
    p_next_sym = p_sym - h * sin(q_sym);
    q_next_sym = q_sym + h * p_next_sym;
    q_sym = q_next_sym;
    p_sym = p_next_sym;
    H_sym(i+1) = calc_H(q_sym, p_sym);
end

% ---------------------------------------------------------
% 3. Heun with Relaxation (Dynamic Time Tracking)
% ---------------------------------------------------------
q_curr = q_init;
p_curr = p_init;
t_current = 0;
tol_time = 1e-12;

for i = 1:N_steps-1
    t_target = t_vec(i+1);
    
    % Dynamic time-tracking loop to reach exact target time
    while (t_target - t_current) > tol_time
        t_remain = t_target - t_current;
        
        if t_remain < h
            dt = t_remain;
        else
            dt = h;
        end
        
        % Stage 1 & 2 derivatives
        k1_q = p_curr;              
        k1_p = -sin(q_curr);
        
        q_s2 = q_curr + dt * k1_q;  
        p_s2 = p_curr + dt * k1_p;
        
        k2_q = p_s2;             
        k2_p = -sin(q_s2);
        
        % Baseline direction vector
        d_q = 0.5 * (k1_q + k2_q);
        d_p = 0.5 * (k1_p + k2_p);
        
        % Solve for gamma
        gamma = solve_relaxation_gamma(q_curr, p_curr, d_q, d_p, dt, H_true);
        
        % Update State (gamma remains active)
        q_curr = q_curr + gamma * dt * d_q;
        p_curr = p_curr + gamma * dt * d_p;
        
        % Update Physical Time
        t_current = t_current + gamma * dt;
    end
    
    % Store exact energy at evaluated temporal boundary
    H_hr(i+1) = calc_H(q_curr, p_curr);
end

% ---------------------------------------------------------
% 4. Integrate Boundary using ode45 (High Precision)
% ---------------------------------------------------------
options = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
[~, y_out_ode45] = ode45(f, t_vec, [q_init; p_init], options);
H_ode45 = calc_H(y_out_ode45(:,1), y_out_ode45(:,2))';

% ---------------------------------------------------------
% 5. Plot Absolute Error in Hamiltonian vs Time
% ---------------------------------------------------------
figure;
hold on;

% Calculate the absolute error |H(t) - H(0)| for all methods
err_euler = abs(H_euler - H_true);
err_heun  = abs(H_heun - H_true);
err_rk4   = abs(H_rk4 - H_true);
err_sv    = abs(H_sv - H_true);
err_sym   = abs(H_sym - H_true);
err_hr    = abs(H_hr - H_true);
err_ode45 = abs(H_ode45 - H_true);

% Plot the active methods with specified colors
% plot(t_vec, err_euler, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Explicit Euler');
plot(t_vec, err_heun, 'r-', 'LineWidth', 2, 'DisplayName', 'Standard Heun');
semilogy(t_vec, err_hr, 'y-', 'LineWidth', 2, 'DisplayName', 'Heun with Relaxation');
% plot(t_vec, err_rk4, 'c-', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
semilogy(t_vec, err_sv, 'g-', 'LineWidth', 2, 'DisplayName', 'Stormer-Verlet');
% plot(t_vec, err_sym, 'm-', 'LineWidth', 2, 'DisplayName', 'Symplectic Euler');
% plot(t_vec, err_ode45, 'w--', 'LineWidth', 2, 'DisplayName', 'ode45');

xlabel('Time (t)');
ylabel('Absolute Energy Error |H(t) - H(0)|');
title('Absolute Error in Energy Conservation over Time');
legend('show', 'Location', 'northwest');

% Formatting to maintain your black background preference
%set(gca, 'YScale', 'log', 'Color', 'k', 'GridColor', 'w'); 
grid on;
hold off;

% ---------------------------------------------------------
% Local Function: Scalar Clamped Newton-Raphson Solver
% ---------------------------------------------------------
function gamma = solve_relaxation_gamma(q_n, p_n, d_q, d_p, dt, H_true)
    gamma = 1.0; 
    tol = 1e-12;
    
    for iter = 1:30
        q_next = q_n + gamma * dt * d_q;
        p_next = p_n + gamma * dt * d_p;
        
        G = 0.5 * p_next^2 - cos(q_next) - H_true;
        
        if abs(G) < tol
            return;
        end
        
        dG = dt * (sin(q_next) * d_q + p_next * d_p);
        
        if abs(dG) < 1e-14
            gamma = 1.0;
            return;
        end
        
        step = G / dG;
        step = max(min(step, 0.5), -0.5); 
        
        gamma = gamma - step;
        
        if gamma < 0.2 || gamma > 1.8
            gamma = 1.0;
            return;
        end
    end
end