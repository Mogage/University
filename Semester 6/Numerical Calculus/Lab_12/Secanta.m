function approx = Secanta(f, approx0, approx1, eps, max_iter)
    q0 = f(approx0);
    q1 = f(approx1);
    prev0 = approx0;
    prev1 = approx1;
    for i=2:max_iter
        x = prev1 - q1 * (prev1 - prev0) / (q1 - q0);
        if abs(x - prev1) < eps
            approx = x;
            return;
        end
        prev0 = prev1;
        q0 = q1;
        prev1 = x;
        q1 = f(x);
    end
    error('Iterations exceeded');
end