% efectuare schimbare variabila
% x -> pi/4*(1+t)
% 0 -> -1
% pi/2 -> 1

f = @(x) (pi/4)./sqrt(sin((pi/4).*(1+x)));
% n = 50;
% [nodes, weights] = GaussQuadrature(n, 'jacobi', 0, 0);
% disp(weights * f(nodes));
[n, value] = compute_quad(f, 1e-9, 'jacobi', 0, 0);
disp(value);
f = @(x) 1./sqrt(sin(x));
disp(quad(f, 0, pi/2));
disp(integral(f, 0, pi/2));
