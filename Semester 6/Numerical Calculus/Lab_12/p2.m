f = @(x)[9 * x(1)^2 + 36 * x(2)^2 + 4 * x(3)^2 - 36;
         x(1)^2 - 2 * x(2)^2 - 20 * x(3);
         x(1)^2 - x(2)^2 + x(3)^2];

f_deriv = @(x)[18 * x(1) + 36 * x(2)^2 + 4 * x(3)^2, 9 * x(1)^2 + 72 * x(2) + 4 * x(3)^2, 9 * x(1)^2 + 36 * x(2)^2 + 8 * x(3);
          2 * x(1) - 2 * x(2)^2 - 20 * x(3), x(1)^2 - 4 * x(2) - 20 * x(3), x(1)^2 - 2 * x(2)^2 - 20;
          2 * x(1) - x(2)^2 + x(3)^2, x(1)^2 - 2 * x(2) + x(3)^2, x(1)^2 - x(2)^2 + 2 * x(3)];

x0 = [1; 1; 0];

approx = Newton(f, f_deriv, 1e-9, x0, 100);
fprintf("Prin metoda lui Newton: ");
fprintf("x = %f, y = %f, z = %f\n", approx);

approx = SuccesiveApprox(f, f_deriv, 1e-9, x0, 100);
fprintf("Prin metoda aproximărilor succesive: ");
fprintf("x = %f, y = %f, z = %f\n", approx);