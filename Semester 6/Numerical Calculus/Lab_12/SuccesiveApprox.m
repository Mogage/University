function approx = SuccesiveApprox(f, f_deriv, eps, x0, max_iter)
    lambda = f_deriv(x0);
    fi = @(x) (x - lambda \ f(x));
    prev = x0(:);
    for k=1:max_iter
        x = fi(prev);
        if abs(prev - x) < eps
            approx = x;
            return
        end
        prev = x;
    end
    error('Iterations exceeded');
end