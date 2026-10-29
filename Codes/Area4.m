% Set the step size and time span
h = 0.5; 
t_end = 100;
t_vec = 0:h:t_end;
N_steps = length(t_vec);

% Vectorized anonymous function for basic RK4/Gauss-Legendre
f = @(t,y) [y(2,:); -sin(y(1,:))]; 

% ---------------------------------------------------------
% 1. Define Initial Square Area (q in [-1,1], p in [0.5,1.5])
% ---------------------------------------------------------
m = 50000;
a = -1;
b = 1;
c = 0.5;
d = 1.5;
q_init = [linspace(a,b,m), b*ones(1,m), linspace(b,a,m), a*ones(1,m)];
p_init = [c*ones(1,m), linspace(c,d,m), d*ones(1,m), linspace(d,c,m)];
num_pts = length(q_init);

% State variables for integrators
y_rk4 = [q_init; p_init];
y_gl4 = [q_init; p_init];

% Initialize arrays to store the area at each time step
area_rk4  = zeros(1, N_steps);
area_gl4  = zeros(1, N_steps);
area_rrk4 = zeros(1, N_steps); % RK4 with Relaxation

initial_area = polyarea(q_init, p_init);
area_rk4(1)  = initial_area;
area_gl4(1)  = initial_area;
area_rrk4(1) = initial_area;

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
% 2. Integrate Boundaries (Standard & Gauss-Legendre Fixed-Time)
% ---------------------------------------------------------
for i = 1:N_steps-1
    ti = t_vec(i);
    
    % --- Standard RK4 Update ---
    k1_rk4 = f(ti, y_rk4);
    k2_rk4 = f(ti + h/2, y_rk4 + h .* k1_rk4 / 2);
    k3_rk4 = f(ti + h/2, y_rk4 + h .* k2_rk4 / 2);
    k4_rk4 = f(ti + h, y_rk4 + h .* k3_rk4);
    y_rk4 = y_rk4 + h .* (k1_rk4 + 2.*k2_rk4 + 2.*k3_rk4 + k4_rk4) / 6;
    area_rk4(i+1) = polyarea(y_rk4(1,:), y_rk4(2,:));
    
    % --- 2-Stage Gauss-Legendre 4 Update (Vectorized) ---
    wi = y_gl4;
    k1_gl = f(ti, wi);
    k2_gl = k1_gl; 
    
    for iter = 1:20
        Y1 = wi + h .* (a11 .* k1_gl + a12 .* k2_gl);
        Y2 = wi + h .* (a21 .* k1_gl + a22 .* k2_gl);
        
        k1_new = f(ti + c1 * h, Y1);
        k2_new = f(ti + c2 * h, Y2);
        
        if max(max(abs(k1_new - k1_gl)), max(abs(k2_new - k2_gl))) < 1e-12
            k1_gl = k1_new;
            k2_gl = k2_new;
            break;
        end
        k1_gl = k1_new;
        k2_gl = k2_new;
    end
    y_gl4 = wi + h .* (b1 .* k1_gl + b2 .* k2_gl);
    area_gl4(i+1) = polyarea(y_gl4(1,:), y_gl4(2,:));
end

% ---------------------------------------------------------
% 3. RK4 with Relaxation (Individual Dynamic Time Tracking)
% ---------------------------------------------------------
q_rrk4_history = zeros(N_steps, num_pts);
p_rrk4_history = zeros(N_steps, num_pts);
q_rrk4_history(1, :) = q_init;
p_rrk4_history(1, :) = p_init;

tol_time = 1e-12;
direction = 1; 

for k = 1:num_pts
    % Isolate point
    q_curr = q_init(k);
    p_curr = p_init(k);
    t_current = 0;
    
    % Set Global Reference Energy for this exact point
    H_true = 0.5 * p_curr^2 - cos(q_curr);
    
    % Iterate over specific target times
    for i = 1:N_steps-1
        t_target = t_vec(i+1);
        
        % Dynamic time-tracking loop
        while direction * (t_target - t_current) > tol_time
            t_remain = t_target - t_current;
            
            if abs(t_remain) < abs(h)
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
            
            % Update State
            q_curr = q_curr + gamma * dt * d_q;
            p_curr = p_curr + gamma * dt * d_p;
            
            % Update Physical Time
            t_current = t_current + gamma * dt;
        end
        
        % Store exact state at evaluated temporal boundary
        q_rrk4_history(i+1, k) = q_curr;
        p_rrk4_history(i+1, k) = p_curr;
    end
end

% Calculate Area for RK4 with Relaxation
for i = 2:N_steps
    area_rrk4(i) = polyarea(q_rrk4_history(i,:), p_rrk4_history(i,:));
end

% ---------------------------------------------------------
% 4. Plot Absolute Error vs Time
% ---------------------------------------------------------
figure;
hold on;

err_rk4   = abs(area_rk4 - initial_area);
err_gl4   = abs(area_gl4 - initial_area);
err_rrk4  = abs(area_rrk4 - initial_area);

plot(t_vec, err_rk4, 'c-', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
plot(t_vec, err_rrk4, 'y-', 'LineWidth', 2, 'DisplayName', 'RK4 with Relaxation');
plot(t_vec, err_gl4, 'g-', 'LineWidth', 2, 'DisplayName', 'Gauss-Legendre 4');

xlabel('Time (t)');
ylabel('Absolute Area Error |A(t) - A(0)|');
title('Absolute Error in Area Conservation over Time');
legend('show', 'Location', 'northwest');

set(gca, 'Color', 'k', 'GridColor', 'w'); 
grid on;
hold off;

% ---------------------------------------------------------
% 4. Plot Absolute Error vs Time (Log Scale)
% ---------------------------------------------------------
figure;
hold on;

err_rk4   = abs(area_rk4 - initial_area);
err_gl4   = abs(area_gl4 - initial_area);
err_rrk4  = abs(area_rrk4 - initial_area);

% Prevent log(0) at t=0 by adding eps (machine epsilon) to the errors
err_rk4_log   = max(err_rk4, eps);
err_gl4_log   = max(err_gl4, eps);
err_rrk4_log  = max(err_rrk4, eps);

plot(t_vec, err_rk4_log, 'c-', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
plot(t_vec, err_rrk4_log, 'y-', 'LineWidth', 2, 'DisplayName', 'RK4 with Relaxation');
plot(t_vec, err_gl4_log, 'g-', 'LineWidth', 2, 'DisplayName', 'Gauss-Legendre 4');

xlabel('Time (t)');
ylabel('Absolute Area Error |A(t) - A(0)| (Log Scale)');
title('Absolute Error in Area Conservation over Time');
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