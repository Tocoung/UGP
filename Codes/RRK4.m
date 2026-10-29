% Set the step size for the Relaxed RK4 method
h = 0.5; 
figure;
hold on;
f=@(t,y) [y(2);-sin(y(1))];

% ---------------------------------------------------------
% 1. Plot the Background Phase Portrait
% ---------------------------------------------------------
for p0=0.2:0.1:1.9
    [t_out, y_out] = rk4_relaxation(f, [0, 15], [0; p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    [t_out, y_out] = rk4_relaxation(f, [0, 15], [0; -p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
for p0=2.1:0.1:3.4
    [t_out, y_out] = rk4_relaxation(f, [0, 15], [0; p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out] = rk4_relaxation(f, [0, -15], [0; p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    [t_out, y_out] = rk4_relaxation(f, [0, 15], [0; -p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out] = rk4_relaxation(f, [0, -15], [0; -p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% ---------------------------------------------------------
% 2. Define Initial Square Area (q in [-1,1], p in [0.5,1.5])
% ---------------------------------------------------------
m=500;
a=-1;
b=1;
c=0.5;
d=1.5;
q_init=[linspace(a,b,m), b*ones(1,m), linspace(b,a,m), a*ones(1,m)];
p_init=[c*ones(1,m), linspace(c,d,m), d*ones(1,m), linspace(d,c,m)];

% Calculate the initial geometric area
area_init = polyarea(q_init, p_init);
fill(q_init, p_init, 'w', 'FaceAlpha', 0.3, 'EdgeColor', 'w', 'LineStyle', '--', ...
    'DisplayName', sprintf('Initial Area (t=0) = %.4f', area_init));

% ---------------------------------------------------------
% 3. Integrate Area Boundary to Specific Time Stamps
% ---------------------------------------------------------
t_stamp = 2; 
q_final=[];
p_final=[];
for k=1:length(q_init)
    [t_out, y_out] = rk4_relaxation(f, [0, t_stamp], [q_init(k); p_init(k)], h);
    q_final=[q_final, y_out(end, 1)];
    p_final=[p_final, y_out(end, 2)];
end
area_final = polyarea(q_final, p_final);
fill(q_final, p_final, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'r', ...
    'DisplayName', sprintf('Deformed Area (t=%g) = %.4f', t_stamp, area_final));

t_stamp = 5; 
q_final=[];
p_final=[];
for k=1:length(q_init)
    [t_out, y_out] = rk4_relaxation(f, [0, t_stamp], [q_init(k); p_init(k)], h);
    q_final=[q_final, y_out(end, 1)];
    p_final=[p_final, y_out(end, 2)];
end
area_final = polyarea(q_final, p_final);
fill(q_final, p_final, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'r', ...
    'DisplayName', sprintf('Deformed Area (t=%g) = %.4f', t_stamp, area_final));

% ---------------------------------------------------------
% 4. Formatting
% ---------------------------------------------------------
xlabel('q (Position)');
ylabel('p (Momentum)');
title('Phase Space Area Evolution (RK4 with Relaxation)');
xlim([-pi pi]);
ylim([-3.5 3.5]);
legend('show', 'Location', 'southwest');
set(gca, 'Color', 'k'); 
hold off;

% ---------------------------------------------------------
% Local Function: RK4 with Relaxation (Time-Tracked)
% ---------------------------------------------------------
function [t_out, y_out] = rk4_relaxation(f, tspan, y0, h)
    t_start = tspan(1);
    t_target = tspan(2);
    
    % Handle forward vs backward integration logically
    if t_target < t_start
        h = -abs(h);
        direction = -1;
    else
        h = abs(h);
        direction = 1;
    end
    
    % Initialize trajectory arrays
    t_out = t_start;
    y_out = y0';
    
    % Set Global Reference Energy 
    H_true = 0.5 * y0(2)^2 - cos(y0(1));
    
    t_current = t_start;
    y_current = y0(:);
    
    tol_time = 1e-12;
    
    % Dynamic time-tracking loop matching integration direction
    while direction * (t_target - t_current) > tol_time
        t_remain = t_target - t_current;
        
        % Set nominal step size (clamped to remaining time on final step)
        if abs(t_remain) < abs(h)
            dt = t_remain;
        else
            dt = h;
        end
        
        q_n = y_current(1);
        p_n = y_current(2);
        
        % Stage 1
        k1 = f(t_current, [q_n; p_n]);
        
        % Stage 2
        y_stage2 = [q_n; p_n] + 0.5 * dt * k1;
        k2 = f(t_current + 0.5 * dt, y_stage2);
        
        % Stage 3
        y_stage3 = [q_n; p_n] + 0.5 * dt * k2;
        k3 = f(t_current + 0.5 * dt, y_stage3);
        
        % Stage 4
        y_stage4 = [q_n; p_n] + dt * k3;
        k4 = f(t_current + dt, y_stage4);
        
        % Baseline RK4 direction vector d_n
        d_vec = (k1 + 2*k2 + 2*k3 + k4) / 6;
        d_q = d_vec(1);
        d_p = d_vec(2);
        
        % Solve relaxation parameter for this exact dt
        gamma = solve_relaxation_gamma(q_n, p_n, d_q, d_p, dt, H_true);
        
        % Update State (gamma is preserved)
        q_next = q_n + gamma * dt * d_q;
        p_next = p_n + gamma * dt * d_p;
        
        % Update Physical Time
        t_current = t_current + gamma * dt;
        y_current = [q_next; p_next];
        
        % Store
        t_out = [t_out; t_current];
        y_out = [y_out; y_current'];
    end
end

% ---------------------------------------------------------
% Local Function: Clamped Newton-Raphson Solver
% ---------------------------------------------------------
function gamma = solve_relaxation_gamma(q_n, p_n, d_q, d_p, dt, H_true)
    gamma = 1.0; % Initial guess from baseline scheme
    tol = 1e-12;
    
    for iter = 1:30
        q_next = q_n + gamma * dt * d_q;
        p_next = p_n + gamma * dt * d_p;
        
        % Residual: G(gamma) = H(y_next) - H_true
        G = 0.5 * p_next^2 - cos(q_next) - H_true;
        
        if abs(G) < tol
            return;
        end
        
        % Derivative: dG/d(gamma)
        dG = dt * (sin(q_next) * d_q + p_next * d_p);
        
        % If derivative vanishes (orthogonal projection), fallback gracefully
        if abs(dG) < 1e-14
            gamma = 1.0;
            return;
        end
        
        % Calculate raw Newton step
        step = G / dG;
        
        % CLAMP THE STEP: Prevent explosions near singularities
        step = max(min(step, 0.5), -0.5); 
        gamma = gamma - step;
        
        % CLAMP GAMMA: Keep the relaxation parameter physically reasonable
        if gamma < 0.2 || gamma > 1.8
            gamma = 1.0;
            return;
        end
    end
end