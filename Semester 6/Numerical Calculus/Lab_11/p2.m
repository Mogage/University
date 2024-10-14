f = @(x) sin(x.^2);
eps = 10e-7;

fprintf("sin(x^2)\n");
[n, value] = compute_quad(f, eps, 'legendre');
fprintf("Legendre: %d noduri necesare. Rezultatul este %f\n", n, value);
fprintf("matlab quad fun: %f\n", quad(f, -1, 1));

f = @(x) cos(x.^2);

fprintf("cos(x^2)\n");
[n, value] = compute_quad(f, eps, 'legendre');
fprintf("Legendre: %d noduri necesare. Rezultatul este %f\n", n, value);
fprintf("matlab quad fun: %f\n", quad(f, -1, 1));
