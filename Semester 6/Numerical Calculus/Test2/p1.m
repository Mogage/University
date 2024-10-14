f = @(x) (sin(x.^2));
f_diff = @(x) (2 .* x .* cos(x.^2));
eval = linspace(-2*pi, 2*pi);
m = 9;
nodes = cos((2 * (0:m) + 1) * pi ./ (2 * m + 2));
t = pi / 5;

% 1.a
plot(eval, f(eval), 'y', 'LineWidth', 2);
hold on;
plot(eval, Lagrange(nodes, f(nodes), eval), 'r--', 'LineWidth', 2);
hold on;
plot(eval, Hermite(nodes, f(nodes), f_diff(nodes), eval), 'b--', 'LineWidth', 2);
hold off;
legend('f(x)', 'Interpolare Lagrange', 'Interpolare Hermite', 'Location', 'best'); 
ylim([-1 1]);

% 1.b
value = f(t);
valueL = Lagrange(nodes, f(nodes), t);
valueH = Hermite(nodes, f(nodes), f_diff(nodes), t);
fprintf("Function value: %d | Lagrange value: %d | Hermite value: %d\n", value, valueL, valueH);

% 1.c
rest_nodes = linspace(min(nodes), max(nodes), 1000);

u = @(x, xi) prod(x-xi);
rest = u(t, nodes) * max(abs(diff(f(rest_nodes), m+1))) / factorial(m+1);
fprintf("Eroare teoretica Lagrange: %d\n", rest);

u = @(x, xi, ri) prod((x-xi).^(2));
rest = u(t, nodes) * max(abs(diff(f(rest_nodes), 2*m+1))) / factorial(2*m+1);
fprintf("Eroare teoretica Hermite: %d\n", rest);

fprintf("Eroare Lagrange: %d | Eroare Hermite: %d\n", value - valueL, value - valueH);
