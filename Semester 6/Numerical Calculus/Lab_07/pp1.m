% nodes = [1.3 1.6 1.9];
% y_values = [0.6200860 0.4554022 0.2818186];
% y_derivatives = [-0.5220232 -0.5698959 -0.5811571];
% eval = 1.5;

nodes = [0 1 2];
y_values = exp(nodes);
y_derivatives = exp(nodes);
eval = [0.25 0.5 0.75];

disp(Hermite(nodes, y_values, y_derivatives, eval));
disp(exp(eval));