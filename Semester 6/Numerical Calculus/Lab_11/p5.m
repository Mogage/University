f = @(x) sin(x);

fprintf("sin(x)\n");
[n, value] = compute_quad(f, 10e-8, 'laguerre', 0);
fprintf("Laguerre: n = %d | res = %f\n", n, value);
f = @(x) exp(-x).*sin(x);
fprintf("matlab quad fun: %f\n", integral(f, 0, inf));

f = @(x) cos(x);

fprintf("cos(x)\n");
[n, value] = compute_quad(f, 10e-8, 'laguerre', 0);
fprintf("Laguerre: n = %d | res = %f\n", n, value);
f = @(x) exp(-x).*cos(x);
fprintf("matlab quad fun: %f\n", integral(f, 0, inf));