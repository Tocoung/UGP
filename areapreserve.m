f=@(t,y) [y(2);-sin(y(1))];

figure;
hold on;
options = odeset(RelTol=1e-2,AbsTol=1e-2);
% ---------------------------------------------------------
% 1. Plot the Background Phase Portrait
% ---------------------------------------------------------
for p0=0.2:0.1:1.9
    [t_out, y_out]=ode45(f, [0, 15], [0; p0], options);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out]=ode45(f, [0, 15], [0; -p0], options);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
for p0=2.1:0.1:3.4
    [t_out, y_out]=ode45(f, [0, 15], [0; p0], options);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out]=ode45(f, [0, -15], [0; p0], options);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    [t_out, y_out]=ode45(f, [0, 15], [0; -p0], options);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    [t_out, y_out]=ode45(f, [0, -15], [0; -p0], options);
    plot(y_out(:,1), y_out(:,2), 'c-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% ---------------------------------------------------------
% 2. Define Initial Square Area (q in [-1,1], p in [2,3])
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
    [t_out, y_out]=ode45(f, [0, t_stamp], [q_init(k); p_init(k)], options);
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
    [t_out, y_out]=ode45(f, [0, t_stamp], [q_init(k); p_init(k)], options);
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
title('Phase Space Area Evolution');
xlim([-pi pi]);
ylim([-3.5 3.5]);
legend('show', 'Location', 'southwest');
set(gca, 'Color', 'k'); 
hold off;