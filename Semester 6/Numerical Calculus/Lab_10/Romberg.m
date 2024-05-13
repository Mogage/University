function res = Romberg(f, a, b)
    eps = 0.00001;
    h = b - a;
    R(1, 1) = h/2*(f(a)+f(b));
    res = R;

    for k=2:25
        R(k, 1) = 0.5 * (R(k-1, 1) + h * sum(f(a + ((1 : 2^(k - 2)) - 0.5) * h)));
        pow4 = 4;
        for j=2:k
            R(k, j) = (pow4*R(k, j-1)-R(k-1, j-1))/(pow4-1);
            pow4 = pow4 * 4;
        end
        if (abs(R(k, k) - R(k-1, k-1)) < eps)
            res = R(k, k);
            return;
        end
        h = h/2;
    end
    res = R(25, 25);
end