% 1.1
disp(["Lagrange 1.1: " Lagrange([3 4 5], [7 6 3], 3.5)]);

% 1.2
plotFundamentals([1 1.3 1.6 1.8 2]);

% 1.3
nodes = [0, 2];
points = 0 : 0.01 : 2;
figure;
plot(points, Lagrange(nodes, exp(nodes), points), 'DisplayName', 'Lagrange', 'LineWidth', 1.5);
hold on;
plot(points, exp(points), 'DisplayName', 'exp', 'LineWidth', 1.5);
legend('show');
hold off;

% 1.4
x = 1.2;
f = @exp;
nodes = [1.1 1.4 1.6];
disp(["Lagrance 1.4: " Lagrange(nodes, f(nodes), x)]);

% 1.5
disp(["Lagrance Bari: " LagrangeBari([3 4 5], [7 6 3], 3.5)]);

% 2.1
t = [1900 1910 1920 1930 1940 1950 1960 1970 1980];
y = [75.995 91.972 105.711 123.203 131.669 150.697 179.323 203.212 226.505];
% disp(Lagrange(t, y, 1975));
disp(["LagrangeBari pop 1975: " LagrangeBari(t, y, 1975)]);
t = [1990 2000 2010];
y = [249.633 281.422 308.786];
% disp(Lagrange(t, y, 2018));
disp(["LagrangeBari pop 2018: " LagrangeBari(t, y, 2018)]);

% 2.2
x = 1.25;
f = @(x)exp(x.^2 - 1);
nodes = [1 1.1 1.2 1.3 1.4];
approx = LagrangeBari(nodes, f(nodes), x);
disp(["LagrangeBari 2.2: " approx]);
% disp(Lagrange(nodes, f(nodes), x));
exact = f(1.25);
disp(["Val exacta: " exact]);
disp(["Eroare: " abs(exact - approx)]);

% 2.3
x = 115;
f = @sqrt;
nodes = [81, 90, 100, 110, 121, 144];
% disp(Lagrange(nodes, f(nodes), x));
disp(["LagrangeBari sqrt(115): " LagrangeBari(nodes, f(nodes), x)]);
disp(["Val exacta: " sqrt(115)]);

% 2.4.a
x = linspace(-5, 5, 1000);
f = @(x) 1 ./ (1 + x.^2);

x_nodes = linspace(-5, 5, 11);
y_nodes = f(x_nodes);
figure;
subplot(2, 1, 1);
plot(x, f(x), 'LineWidth', 2);
hold on;
plot(x, LagrangeBari(x_nodes, y_nodes, x), '--', 'LineWidth', 2);
scatter(x_nodes, y_nodes, 'filled', 'MarkerFaceColor', 'r');
title('Runge - Echidistant');
legend('f(x)', 'Interpolare Lagrange', 'Noduri');
hold off;

x_nodes_cheb = 5*cos(((0:11)*pi)/11);
y_nodes_cheb = f(x_nodes_cheb);
subplot(2, 1, 2);
plot(x, f(x), 'LineWidth', 2);
hold on;
plot(x, LagrangeBari(x_nodes_cheb, y_nodes_cheb, x), '--', 'LineWidth', 2);
scatter(x_nodes_cheb, y_nodes_cheb, 'filled', 'MarkerFaceColor', 'r');
title('Runge - Cebîșev');
legend('f(x)', 'Interpolare Lagrange', 'Noduri');
hold off;

% 2.4.b
x = linspace(-1, 1, 1000);
f = @abs;

x_nodes = linspace(-1, 1, 11);
y_nodes = f(x_nodes);
figure;
subplot(2, 1, 1);
plot(x, f(x), 'LineWidth', 2);
hold on;
plot(x, LagrangeBari(x_nodes, y_nodes, x), '--', 'LineWidth', 2);
scatter(x_nodes, y_nodes, 'filled', 'MarkerFaceColor', 'r');
title('Bernstein - Echidistant');
legend('f(x)', 'Interpolare Lagrange', 'Noduri');
hold off;

x_nodes_cheb = cos(((0:11)*pi)/11);
y_nodes_cheb = f(x_nodes_cheb);
subplot(2, 1, 2);
plot(x, f(x), 'LineWidth', 2);
hold on;
plot(x, LagrangeBari(x_nodes_cheb, y_nodes_cheb, x), '--', 'LineWidth', 2);
scatter(x_nodes_cheb, y_nodes_cheb, 'filled', 'MarkerFaceColor', 'r');
title('Bernstein - Cebîșev');
legend('f(x)', 'Interpolare Lagrange', 'Noduri');
hold off;