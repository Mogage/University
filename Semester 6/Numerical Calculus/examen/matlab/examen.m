% x_k1 = x_k - f(x_k)/sqrt(D(f)(x_k)^2 - f(x_k)*(D@@2)(f)(x_k));  
syms x f(x)
phi = x-f(x)/(sqrt(diff(f(x), x, 1)^2 - f(x)*diff(f(x), x, 2)));
diff(phi, 1)
