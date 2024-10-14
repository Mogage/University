function approx = Newton(f, f_deriv, eps, x0, max_iter)
    prev = x0(:);
    for k=1:max_iter
        x = prev - f_deriv(prev) \ f(prev);
        if norm(x - prev, inf) < eps
            approx = x;
            return
        end
        prev = x;
    end
    error('Iterations exceeded');
end