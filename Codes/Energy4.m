% Set the step size and time span
h = 0.2; 
t_end = 100;
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
y_rk4 = [q_init; p_init]; 
y_gl4 = [q_init; p_init];

% Initialize arrays to store the Hamiltonian at each time step
H_rk4   = zeros(1, N_steps);
H_gl4   = zeros(1, N_steps);
H_rrk4  = zeros(1, N_steps); % RK4 with Relaxation

% Calculate and store initial Hamiltonian
H_rk4(1)  = H_true;
H_gl4(1)  = H_true;
H_rrk4(1) = H_true;

% Butcher Tableau Constants for Gauss-Legendre 4
sq3 = sqrt(3) / 6;
c1  = 0.5 - sq3;
c2  = 0.5 + sq3;
a11 = 0.25;
a12 = 0.25 - sq3;
a21 = 0.25 + sq3;
a22 = 0.25;
b1  = 0.5;
b2  = 0.5;

% ---------------------------------------------------------
% 2. Integrate and Calculate Hamiltonian (Standard Methods)
% ---------------------------------------------------------
for i = 1:N_steps-1
    ti = t_vec(i);
    
    % --- Standard RK4 Update ---
    k1_rk4 = f(ti, y_rk4);
    k2_rk4 = f(ti + h/2, y_rk4 + h * k1_rk4 / 2);
    k3_rk4 = f(ti + h/2, y_rk4 + h * k2_rk4 / 2);
    k4_rk4 = f(ti + h, y_rk4 + h * k3_rk4);
    y_rk4 = y_rk4 + h * (k1_rk4 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4) / 6;
    H_rk4(i+1) = calc_H(y_rk4(1), y_rk4(2));
    
    % --- 2-Stage Gauss-Legendre 4 Update ---
    wi = y_gl4;
    k1_gl = f(ti, wi);
    k2_gl = k1_gl; 
    
    for iter = 1:20
        Y1 = wi + h * (a11 * k1_gl + a12 * k2_gl);
        Y2 = wi + h * (a21 * k1_gl + a22 * k2_gl);
        
        k1_new = f(ti + c1 * h, Y1);
        k2_new = f(ti + c2 * h, Y2);
        
        if max(norm(k1_new - k1_gl), norm(k2_new - k2_gl)) < 1e-12
            k1_gl = k1_new;
            k2_gl = k2_new;
            break;
        end
        k1_gl = k1_new;
        k2_gl = k2_new;
    end
    y_gl4 = wi + h * (b1 * k1_gl + b2 * k2_gl);
    H_gl4(i+1) = calc_H(y_gl4(1), y_gl4(2));
end

% ---------------------------------------------------------
% 3. RK4 with Relaxation (Dynamic Time Tracking)
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
        
        % Stage 1 to 4 derivatives (RK4)
        k1_q = p_curr;              
        k1_p = -sin(q_curr);
        
        q_s2 = q_curr + 0.5 * dt * k1_q;  
        p_s2 = p_curr + 0.5 * dt * k1_p;
        k2_q = p_s2;             
        k2_p = -sin(q_s2);
        
        q_s3 = q_curr + 0.5 * dt * k2_q;
        p_s3 = p_curr + 0.5 * dt * k2_p;
        k3_q = p_s3;
        k3_p = -sin(q_s3);
        
        q_s4 = q_curr + dt * k3_q;
        p_s4 = p_curr + dt * k3_p;
        k4_q = p_s4;
        k4_p = -sin(q_s4);
        
        % Baseline direction vector
        d_q = (k1_q + 2*k2_q + 2*k3_q + k4_q) / 6;
        d_p = (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6;
        
        % Solve for gamma
        gamma = solve_relaxation_gamma(q_curr, p_curr, d_q, d_p, dt, H_true);
        
        % Update State (gamma remains active)
        q_curr = q_curr + gamma * dt * d_q;
        p_curr = p_curr + gamma * dt * d_p;
        
        % Update Physical Time
        t_current = t_current + gamma * dt;
    end
    
    % Store exact energy at evaluated temporal boundary
    H_rrk4(i+1) = calc_H(q_curr, p_curr);
end

% ---------------------------------------------------------
% 4. Plot Absolute Error in Hamiltonian vs Time (Log Scale)
% ---------------------------------------------------------
figure;
hold on;

% Calculate the absolute error |H(t) - H(0)| 
err_rk4  = abs(H_rk4 - H_true);
err_gl4  = abs(H_gl4 - H_true);
err_rrk4 = abs(H_rrk4 - H_true);

% Prevent log(0) at t=0 by adding eps (machine epsilon) to the errors
err_rk4_log  = max(err_rk4, eps);
err_gl4_log  = max(err_gl4, eps);
err_rrk4_log = max(err_rrk4, eps);

% Plot the active methods with specified colors
plot(t_vec, err_rk4_log, 'c-', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
plot(t_vec, err_rrk4_log, 'y-', 'LineWidth', 2, 'DisplayName', 'RK4 with Relaxation');
plot(t_vec, err_gl4_log, 'g-', 'LineWidth', 2, 'DisplayName', 'Gauss-Legendre 4');

xlabel('Time (t)');
ylabel('Absolute Energy Error |H(t) - H(0)| (Log Scale)');
title('Absolute Error in Energy Conservation over Time');
legend('show', 'Location', 'northwest');

grid on;
hold off;


figure;
hold on;

% Calculate the absolute error |H(t) - H(0)| 
err_rk4  = abs(H_rk4 - H_true);
err_gl4  = abs(H_gl4 - H_true);
err_rrk4 = abs(H_rrk4 - H_true);

% Prevent log(0) at t=0 by adding eps (machine epsilon) to the errors
err_rk4_log  = max(err_rk4, eps);
err_gl4_log  = max(err_gl4, eps);
err_rrk4_log = max(err_rrk4, eps);

% Plot the active methods with specified colors
plot(t_vec, err_rk4_log, 'c-', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
plot(t_vec, err_rrk4_log, 'y-', 'LineWidth', 2, 'DisplayName', 'RK4 with Relaxation');
plot(t_vec, err_gl4_log, 'g-', 'LineWidth', 2, 'DisplayName', 'Gauss-Legendre 4');

xlabel('Time (t)');
ylabel('Absolute Energy Error |H(t) - H(0)| (Log Scale)');
title('Absolute Error in Energy Conservation over Time');
legend('show', 'Location', 'northwest');

% Explicitly force the Y-axis to be logarithmic
set(gca, 'YScale', 'log', 'Color', 'k', 'GridColor', 'w'); 
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