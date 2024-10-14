% Facem schimbare de variabila t = 3/2 + 1/2*x
% Aceasta vine de la (b+a)/2 + (b-a)/2 * x
% Ulterior, integrala devine sqrt(1-x^2)*1/4*sin(3/2+1/2*x)

f = @(x) 0.25 * sin(1.5 + 0.5*x);
[n, value] = compute_quad(f, 1e-10, 'chebyshev2');
disp(value);

% Pentru verificare, folosim functia quad din matlab sa verificam
% rezultatul
f = @(x) sqrt(3.*x-x.^2-2).*sin(x);
disp(quad(f, 1, 2));