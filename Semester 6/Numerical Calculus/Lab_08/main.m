nodes = [0 1 2.5 3.6 5 7 8.1 10];
values = sin(nodes);
eval = 0:.25:10;

plot(eval, spline(nodes, values, eval), 'b', 'LineWidth', 2);
hold on;
coef = cubicSplinec(nodes, values, 2); % natural
plot(eval, evalSplinec(nodes, coef, eval), 'g--', 'LineWidth', 2);
hold on;
coef = cubicSplinec(nodes, values, 0, [cos(nodes(1)), cos(nodes(end))]); % complet
plot(eval, evalSplinec(nodes, coef, eval), 'r--', 'LineWidth', 2);
hold on;
coef = cubicSplinec(nodes, values, 1, [-sin(nodes(1)), -sin(nodes(end))]); % 2nd deriv
plot(eval, evalSplinec(nodes, coef, eval), 'm--', 'LineWidth', 2);
hold on;
coef = cubicSplinec(nodes, values, 3); % deBoor
plot(eval, evalSplinec(nodes, coef, eval), 'k--', 'LineWidth', 2);
legend('Matlab', 'Natural', 'Complet', '2ndDeriv', 'DeBoor', 'Location', 'best'); 
hold off;

figure;
plot(eval, spline(nodes, values, eval), 'b', 'LineWidth', 2);
hold on;
coef = cubicSpline(nodes, values, 0); % natural
plot(eval, evalSpline(nodes, coef, eval), 'g--', 'LineWidth', 2);
hold on;
coef = cubicSpline(nodes, values, 1, [cos(nodes(1)), cos(nodes(end))]); % complet
plot(eval, evalSpline(nodes, coef, eval), 'r--', 'LineWidth', 2);
hold on;
coef = cubicSpline(nodes, values, 2, [-sin(nodes(1)), -sin(nodes(end))]); % 2nd deriv
plot(eval, evalSpline(nodes, coef, eval), 'm--', 'LineWidth', 2);
hold on;
coef = cubicSpline(nodes, values, 3); % deBoor
plot(eval, evalSpline(nodes, coef, eval), 'k--', 'LineWidth', 2);
legend('Matlab', 'Natural', 'Complet', '2ndDeriv', 'DeBoor', 'Location', 'best'); 
hold off;