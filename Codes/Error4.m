% Set the fixed final time for the convergence evaluation
t_end = 100; 

% Array of step sizes (h) to test (from 0.01 to 0.5)
h_vec = logspace(-2, log10(0.5), 8); 
num_h = length(h_vec);

% Vectorized anonymous functions
f = @(t,y) [y(2); -sin(y(1))]; 

% The Hamiltonian for the normalized mathematical pendulum
calc_H = @(q, p) 0.5 .* p.^2 - cos(q); 

% ---------------------------------------------------------
% 1. Define Initial Point & Exact Reference Solution
% ---------------------------------------------------------
q_init = 0.0; 
p_init = 1.0; 
y0 = [q_init; p_init];
H_true = calc_H(q_init, p_init);

% High-precision reference solution using ode45
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
[~, y_ref] = ode45(f, [0, t_end], y0, options);
y_exact = y_ref(end, :)'; % The "true" state at t = 10

% Initialize arrays to store the global error for each h
err_rk4  = zeros(1, num_h);
err_gl4  = zeros(1, num_h);
err_rrk4 = zeros(1, num_h);

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
% 2. Convergence Loop over Step Sizes
% ---------------------------------------------------------
for idx = 1:num_h
    h = h_vec(idx);
    tol_time = 1e-12;
    
    % --- Standard RK4 Integration ---
    y_rk4 = y0;
    t_curr = 0;
    while (t_end - t_curr) > tol_time
        dt = min(h, t_end - t_curr);
        k1 = f(0, y_rk4);
        k2 = f(0, y_rk4 + 0.5 * dt * k1);
        k3 = f(0, y_rk4 + 0.5 * dt * k2);
        k4 = f(0, y_rk4 + dt * k3);
        y_rk4 = y_rk4 + dt * (k1 + 2*k2 + 2*k3 + k4) / 6;
        t_curr = t_curr + dt;
    end
    err_rk4(idx) = norm(y_rk4 - y_exact);
    
    % --- Gauss-Legendre 4 Integration ---
    y_gl4 = y0;
    t_curr = 0;
    while (t_end - t_curr) > tol_time
        dt = min(h, t_end - t_curr);
        wi = y_gl4;
        k1_gl = f(0, wi);
        k2_gl = k1_gl; 
        
        for iter = 1:20
            Y1 = wi + dt * (a11 * k1_gl + a12 * k2_gl);
            Y2 = wi + dt * (a21 * k1_gl + a22 * k2_gl);
            k1_new = f(0, Y1);
            k2_new = f(0, Y2);
            if max(norm(k1_new - k1_gl), norm(k2_new - k2_gl)) < 1e-12
                k1_gl = k1_new;
                k2_gl = k2_new;
                break;
            end
            k1_gl = k1_new;
            k2_gl = k2_new;
        end
        y_gl4 = wi + dt * (b1 * k1_gl + b2 * k2_gl);
        t_curr = t_curr + dt;
    end
    err_gl4(idx) = norm(y_gl4 - y_exact);
    
    % --- RK4 with Relaxation Integration ---
    q_curr = q_init;
    p_curr = p_init;
    t_curr = 0;
    while (t_end - t_curr) > tol_time
        t_remain = t_end - t_curr;
        if t_remain < h
            dt = t_remain;
        else
            dt = h;
        end
        
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
        
        d_q = (k1_q + 2*k2_q + 2*k3_q + k4_q) / 6;
        d_p = (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6;
        
        gamma = solve_relaxation_gamma(q_curr, p_curr, d_q, d_p, dt, H_true);
        
        q_curr = q_curr + gamma * dt * d_q;
        p_curr = p_curr + gamma * dt * d_p;
        t_curr = t_curr + gamma * dt;
    end
    y_rrk4 = [q_curr; p_curr];
    err_rrk4(idx) = norm(y_rrk4 - y_exact);
end

% ---------------------------------------------------------
% 3. Plot Convergence (Error vs Step Size) on Log-Log Scale
% ---------------------------------------------------------
figure;
hold on;

% Plot the errors for the three methods
loglog(h_vec, err_rk4, 'c-o', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
loglog(h_vec, err_rrk4, 'y-s', 'LineWidth', 2, 'DisplayName', 'RK4 with Relaxation');
loglog(h_vec, err_gl4, 'g-^', 'LineWidth', 2, 'DisplayName', 'Gauss-Legendre 4');

% Create a mathematical O(h^4) reference line to visually verify the slope
ref_line = (err_rk4(end) / h_vec(end)^4) * h_vec.^4;
loglog(h_vec, ref_line, 'w--', 'LineWidth', 1.5, 'DisplayName', 'Reference Slope: O(h^4)');

xlabel('Step Size (h)');
ylabel('Absolute Solution Error at t=10 |y(t) - y_{exact}|');
title('Convergence Study: Global Error vs. Step Size');
legend('show', 'Location', 'northwest');

set(gca, 'XScale', 'log', 'YScale', 'log', 'Color', 'k', 'GridColor', 'w');
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