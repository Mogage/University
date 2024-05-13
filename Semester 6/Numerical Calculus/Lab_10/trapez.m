function res = trapez(f, a, b, n)
    h = (b - a) / n;
    k = 0 : n;
    x = a + k * h;

    sum = 0;
    for k = 2 : n - 1
        sum = sum + f(x(k));
    end

    res = ((b - a) / (2 * n)) * (f(a) + f(b) + 2 * sum);
end
