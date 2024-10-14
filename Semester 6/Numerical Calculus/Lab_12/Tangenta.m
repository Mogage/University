function approx = Tangenta(f, f_deriv, approx0, eps, max_iter)
    prev = approx0;
    for i=1:max_iter
        x = prev - f(prev) / f_deriv(prev);
        if abs(x - prev) < eps
            approx = x;
            return;
        end
        prev = x;
    end
    error('Iterations exceeded');
end