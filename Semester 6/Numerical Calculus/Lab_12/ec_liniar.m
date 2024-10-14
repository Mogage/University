f = @(x) x.^2 + sin(x) + exp(-x) - 2;
f_deriv = @(x) 2.*x - exp(-x) + cos(x);

disp(Tangenta(f, f_deriv, 0.5, 1e-5, 100));
disp(Secanta(f, 0.5, 1, 1e-5, 100));