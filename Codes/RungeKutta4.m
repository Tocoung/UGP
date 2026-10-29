% Set the step size for the RK4 method
h = 0.5; 
figure;
hold on;

f=@(t,y) [y(2);-sin(y(1))];

% ---------------------------------------------------------
% 1. Plot the Background Phase Portrait
% ---------------------------------------------------------
for p0=0.2:0.1:1.9
    [t_out, y_out] = rk4_method(f, [0, 15], [0; p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    [t_out, y_out] = rk4_method(f, [0, 15], [0; -p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
for p0=2.1:0.1:3.4
    [t_out, y_out] = rk4_method(f, [0, 15], [0; p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out] = rk4_method(f, [0, -15], [0; p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    [t_out, y_out] = rk4_method(f, [0, 15], [0; -p0], h);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out] = rk4_method(f, [0, -15], [0; -p0], h);
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

% Fill the initial square with semi-transparent white
fill(q_init, p_init, 'w', 'FaceAlpha', 0.3, 'EdgeColor', 'w', 'LineStyle', '--', ...
    'DisplayName', sprintf('Initial Area (t=0) = %.4f', area_init));

% ---------------------------------------------------------
% 3. Integrate Area Boundary to a Single Time Stamp
% ---------------------------------------------------------
t_stamp = 2; % Set your single time stamp here
q_final=[];
p_final=[];

for k=1:length(q_init)
    [t_out, y_out] = rk4_method(f, [0, t_stamp], [q_init(k); p_init(k)], h);
    q_final=[q_final, y_out(end, 1)];
    p_final=[p_final, y_out(end, 2)];
end

% Calculate the final geometric area
area_final = polyarea(q_final, p_final);

% Fill the deformed shape with semi-transparent red
fill(q_final, p_final, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'r', ...
    'DisplayName', sprintf('Deformed Area (t=%g) = %.4f', t_stamp, area_final));

t_stamp = 5; % Set your single time stamp here
q_final=[];
p_final=[];

for k=1:length(q_init)
    [t_out, y_out] = rk4_method(f, [0, t_stamp], [q_init(k); p_init(k)], h);
    q_final=[q_final, y_out(end, 1)];
    p_final=[p_final, y_out(end, 2)];
end

% Calculate the final geometric area
area_final = polyarea(q_final, p_final);

% Fill the deformed shape with semi-transparent red
fill(q_final, p_final, 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'r', ...
    'DisplayName', sprintf('Deformed Area (t=%g) = %.4f', t_stamp, area_final));

% ---------------------------------------------------------
% 4. Formatting
% ---------------------------------------------------------
xlabel('q (Position)');
ylabel('p (Momentum)');
title('Phase Space Area Evolution (Standard RK4)');
xlim([-pi pi]);
ylim([-3.5 3.5]);
legend('show', 'Location', 'southwest');
set(gca, 'Color', 'k'); 
hold off;

% ---------------------------------------------------------
% Local Function: Standard RK4 Integrator
% ---------------------------------------------------------
function [t_out, y_out] = rk4_method(f, tspan, y0, h)
    % Handle forward vs backward integration
    if tspan(2) < tspan(1)
        h = -abs(h);
    else
        h = abs(h);
    end
    
    % Generate time vector ensuring we hit the exact final time
    t_out = tspan(1):h:tspan(2);
    if t_out(end) ~= tspan(2)
        t_out = [t_out, tspan(2)];
    end
    
    N = length(t_out);
    y_out = zeros(N, 2);
    y_out(1, :) = y0';
    
    for i = 1:N-1
        dt = t_out(i+1) - t_out(i); 
        ti = t_out(i);
        wi = y_out(i, :)';
        
        k1=f(ti,wi);
        k2=f(ti+dt/2,wi+dt.*k1/2);
        k3=f(ti+dt/2,wi+dt.*k2/2);
        k4=f(ti+dt,wi+dt.*k3);
        
        wi_next=wi+dt.*(k1+2.*k2+2.*k3+k4)/6;
        
        y_out(i+1, 1) = wi_next(1);
        y_out(i+1, 2) = wi_next(2);
    end
end