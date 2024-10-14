f = @(x) [x(1)^2 + x(2)^2 - 1;
          x(1)^3 - x(2)];
f_deriv = @(x) [2 * x(1), 2 * x(2);
                3 * x(1) ^ 2, -1];

x0 = [1; 1];
approx = Newton(f, f_deriv, 1e-9, x0, 100);
disp(approx);
approx = SuccesiveApprox(f, f_deriv, 1e-9, x0, 100);
disp(approx);