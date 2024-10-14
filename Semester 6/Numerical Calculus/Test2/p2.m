eval = linspace(-2 * pi, 2 * pi);
points = -2 * pi : 0.25 : 2 * pi;
f = @(x) (x.^2 .* sin(x));

% 2.a

f_diff = @(x) (2 .* x .* sin(x) + x .^ 2 .* cos(x));
m = 12;
nodes = cos((0:m)*pi./m);
nodes = sort(2 * pi * nodes); % scalare 

figure;
subplot(2, 1, 1);
hold on;
values = evalSplinec(nodes, cubicSplinec(nodes, f(nodes), 3, f_diff(nodes)), points);
plot(points, values, 'b', 'LineWidth', 2);
plot(points, f(points), 'r--', 'LineWidth', 2);
legend("Spline Ceb", "f(x)");
hold off;

subplot(2, 1, 2);
hold on;
values = evalSplinec(eval, cubicSplinec(eval, f(eval), 3, f_diff(points)), points);
plot(points, values, 'b', 'LineWidth', 2);
plot(points, f(points), 'r--', 'LineWidth', 2);
legend("Spline Echi", "f(x)");
hold off;

% 2.b

n = 11;
values = mcmmp(eval, f(eval), n, points);
% base = @(x)[ones(1, length(x)); x; x.^2; x.^3; x.^4; x.^5; x.^6; x.^7; x.^8; x.^9; x.^10]; % n=11
% values = mcmmp(nodes, f(nodes), base, eval);
figure;
hold on;
plot(points, values, 'b', 'LineWidth', 2);
plot(points, f(points), 'r--', 'LineWidth', 2);
legend("MCMMP", "f(x)");
hold off;