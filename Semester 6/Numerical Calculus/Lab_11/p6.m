f = @(x) sin(x);
n = 3;

fprintf("sin(x)\n");
[nodes, weights] = GaussQuadrature(n, 'hermite');
fprintf("Hermite: %f\n", weights * f(nodes));
f = @(x) exp(-(x.^2)).*sin(x);
fprintf("matlab integral fun: %f\n", integral(f, -inf, inf));

f = @(x) cos(x);
n = 3;

fprintf("cos(x)\n");
[nodes, weights] = GaussQuadrature(n, 'hermite');
fprintf("Hermite: %f\n", weights * f(nodes));
f = @(x) exp(-(x.^2)).*cos(x);
fprintf("matlab integral fun: %f\n", integral(f, -inf, inf));