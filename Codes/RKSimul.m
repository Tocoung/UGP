f=@(t,y) [-y(2),y(1)];
figure;
hold on;
h_i=[];
e_i=[];
for j=2:5
    ti=0;
    wi=[0,1];
    h=2^-j;
    t_i=ti;
    w_i=1;
    for i=1:20*2^j
        k1=f(ti,wi);
        k2=f(ti+h/2,wi+h.*k1/2);
        k3=f(ti+h/2,wi+h.*k2/2);
        k4=f(ti+h,wi+h.*k3);
        wi=wi+h.*(k1+2.*k2+2.*k3+k4)/6;
        ti=ti+h;
        t_i=[t_i,ti];
        w_i=[w_i,wi(2)];
    end
    plot(t_i, w_i, '-', 'DisplayName', sprintf('h = 2^{-%d}', j));
    e=sqrt(mean((w_i-cos(t_i)).^2));
    h_i=[h_i,h];
    e_i=[e_i,e];
end
t_actual=linspace(0,20,10000);
plot(t_actual, cos(t_actual), 'w--', 'LineWidth', 1.5, 'DisplayName', 'Actual: cos(t)');
xlabel('Time (t)');
ylabel('y_2 (Position)');
title('RK4 Method for Pendulum Equation');
legend('show'); % Shows which line corresponds to which step size
hold off;


figure;
hold on;
loglog(h_i,e_i,'o-','LineWidth',1.5,'DisplayName','RK4 RMSE');
ref_line = e_i(end)*(h_i/h_i(end)).^4; 
loglog(h_i,ref_line,'w--','LineWidth',1.5,'DisplayName','Theoretical Slope = 4');
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Step Size (h)');
ylabel('RMSE');
title('Log-Log Plot of RK4 Error Convergence (RMSE)');
legend('show', 'Location', 'northwest');
hold off;