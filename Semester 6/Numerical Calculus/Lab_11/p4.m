f = @(x) exp(-(x.^2));
n = 10;

[nodes, weights] = GaussQuadrature(n, 'chebyshev2');
fprintf("Chebyshev2: %f\n", weights * f(nodes));
f = @(x) exp(-(x.^2)).*sqrt(1-x.^2);
fprintf("matlab quad fun: %f\n", quad(f, -1, 1));