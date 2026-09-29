% Set the step size and time span
h = 0.2; 
t_end = 100;
t_vec = 0:h:t_end;
N_steps = length(t_vec);

% Vectorized anonymous function for RK4 and Heun
f = @(t,y) [y(2,:); -sin(y(1,:))]; 

% ---------------------------------------------------------
% 1. Define Initial Square Area (q in [-1,1], p in [0.5,1.5])
% ---------------------------------------------------------
m = 1000;
a = -1;
b = 1;
c = 0.5;
d = 1.5;
q_init = [linspace(a,b,m), b*ones(1,m), linspace(b,a,m), a*ones(1,m)];
p_init = [c*ones(1,m), linspace(c,d,m), d*ones(1,m), linspace(d,c,m)];
num_pts = length(q_init);

% State variables for Explicit Euler
q_euler = q_init;
p_euler = p_init;

% State variables for RK4 and Heun (Formatted as 2xM matrices)
y_rk4 = [q_init; p_init];
y_heun = [q_init; p_init];

% State variables for Stormer-Verlet
q_sv = q_init;
p_sv = p_init;

% State variables for Symplectic Euler
q_sym = q_init;
p_sym = p_init;

% Initialize arrays to store the area at each time step
area_euler = zeros(1, N_steps);
area_rk4 = zeros(1, N_steps);
area_heun = zeros(1, N_steps);
area_sv = zeros(1, N_steps);
area_sym = zeros(1, N_steps);

initial_area = polyarea(q_init, p_init);
area_euler(1) = initial_area;
area_rk4(1) = initial_area;
area_heun(1) = initial_area;
area_sv(1) = initial_area;
area_sym(1) = initial_area;

% ---------------------------------------------------------
% 2. Integrate Boundaries and Calculate Areas Step-by-Step
% ---------------------------------------------------------
for i = 1:N_steps-1
    ti = t_vec(i);
    
    % --- Explicit Euler Update ---
    p_next_euler = p_euler - h .* sin(q_euler);
    q_next_euler = q_euler + h .* p_euler; 
    
    q_euler = q_next_euler;
    p_euler = p_next_euler;
    
    area_euler(i+1) = polyarea(q_euler, p_euler);
    
    % --- Heun's Method Update (2nd-Order Explicit RK) ---
    k1_heun = f(ti, y_heun);
    k2_heun = f(ti + h, y_heun + h .* k1_heun);
    
    y_heun = y_heun + h .* (k1_heun + k2_heun) / 2;
    
    area_heun(i+1) = polyarea(y_heun(1,:), y_heun(2,:));
    
    % --- Standard RK4 Update ---
    k1 = f(ti, y_rk4);
    k2 = f(ti + h/2, y_rk4 + h .* k1 / 2);
    k3 = f(ti + h/2, y_rk4 + h .* k2 / 2);
    k4 = f(ti + h, y_rk4 + h .* k3);
    
    y_rk4 = y_rk4 + h .* (k1 + 2.*k2 + 2.*k3 + k4) / 6;
    
    area_rk4(i+1) = polyarea(y_rk4(1,:), y_rk4(2,:));
    
    % --- Stormer-Verlet Update ---
    % Using the explicit Momentum-Verlet formulation
    p_half_sv = p_sv - (h / 2) .* sin(q_sv);
    q_next_sv = q_sv + h .* p_half_sv;
    p_next_sv = p_half_sv - (h / 2) .* sin(q_next_sv);
    
    q_sv = q_next_sv;
    p_sv = p_next_sv;
    
    area_sv(i+1) = polyarea(q_sv, p_sv);
    % --- Symplectic Euler Update ---
    % 1. Update momentum using current position
    p_next_sym = p_sym - h .* sin(q_sym);
    % 2. Update position using the NEW momentum
    q_next_sym = q_sym + h .* p_next_sym;
    
    q_sym = q_next_sym;
    p_sym = p_next_sym;
    
    area_sym(i+1) = polyarea(q_sym, p_sym);
end

% ---------------------------------------------------------
% 3. Integrate Boundary using ode45 (High Precision)
% ---------------------------------------------------------
options = odeset('RelTol', 1e-7, 'AbsTol', 1e-7);
% Create a vectorized ODE function that takes a column vector 
f_ode45 = @(t, y) [y(num_pts+1:end); -sin(y(1:num_pts))];
y0_ode45 = [q_init'; p_init']; 
% Solve for the entire boundary in one shot evaluating exactly at t_vec
[~, y_out_ode45] = ode45(f_ode45, t_vec, y0_ode45, options);

% Calculate area for ode45 at each time step
area_ode45 = zeros(1, N_steps);
for i = 1:N_steps
    q_out = y_out_ode45(i, 1:num_pts);
    p_out = y_out_ode45(i, num_pts+1:end);
    area_ode45(i) = polyarea(q_out, p_out);
end

% ---------------------------------------------------------
% 4. Plot Absolute Error vs Time
% ---------------------------------------------------------
figure;
hold on;

% Calculate the absolute error |A(t) - A(0)| for all methods
err_euler = abs(area_euler - initial_area);
err_heun  = abs(area_heun - initial_area);
err_rk4   = abs(area_rk4 - initial_area);
err_sv    = abs(area_sv - initial_area);
err_sym   = abs(area_sym - initial_area);
err_ode45 = abs(area_ode45 - initial_area);

% Plotting only Heun's Method and Stormer-Verlet as requested
plot(t_vec, err_euler, 'r-', 'LineWidth', 2, 'DisplayName', 'Explicit Euler');
%plot(t_vec, err_heun, 'y-', 'LineWidth', 2, 'DisplayName', 'Heun''s Method');
% plot(t_vec, err_rk4, 'c-', 'LineWidth', 2, 'DisplayName', 'Standard RK4');
%plot(t_vec, err_sv, 'g-', 'LineWidth', 2, 'DisplayName', 'Stormer-Verlet');
plot(t_vec, err_sym, 'm-', 'LineWidth', 2, 'DisplayName', 'Symplectic Euler');
% plot(t_vec, err_ode45, 'w--', 'LineWidth', 2, 'DisplayName', 'ode45');

xlabel('Time (t)');
ylabel('Absolute Area Error |A(t) - A(0)|');
title('Absolute Error in Area Conservation over Time');
legend('show', 'Location', 'northwest');

% Formatting to maintain your black background preference
set(gca, 'Color', 'k', 'GridColor', 'w'); 
grid on;
hold off;