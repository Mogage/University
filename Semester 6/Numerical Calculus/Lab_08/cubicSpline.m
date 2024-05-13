function coef = cubicSpline(nodes, values, type, der)
    n = length(nodes);
    h = diff(nodes);
    coef = zeros(n, 4);
    coef(:, 1) = values';

    a = zeros(1, n);
    if (type == 1 || type == 2)
        a(1) = 3 * ((coef(2, 1) - coef(1, 1)) / h(1) - der(1));
        a(end) = 3 * (der(2) - (coef(end, 1) - coef(end - 1, 1)) / h(end - 1));
    end
    for i=2:n-1
        a(i) = 3*((coef(i+1, 1)-coef(i, 1))/h(i) - (coef(i, 1) - coef(i-1, 1))/h(i-1));
    end
    l = zeros(1, n);
    m = zeros(1, n);
    z = zeros(1, n);
    switch type
        case 0
            l(1) = 1;
        case {1 2}
            l(1) = 2 * h(1);
            m(1) = 0.5;
            z(1) = a(1) / l(1);
        case 3
            l(1) = 2 * h(1);
            m(1) = 1;
            z(1) = a(1) / l(1);
    end

    for i=2:n-1
        l(i) = 2*(nodes(i+1)-nodes(i-1)) - h(i-1)*m(i-1);
        m(i) = h(i)/l(i);
        z(i) = (a(i)-h(i-1)*z(i-1))/l(i);
    end
    
    if (type ~= 0)
        l(n) = h(n-1) * (2 - m(n - 1));
        coef(n, 3) = (a(n)- h(n-1)*z(n-1))/l(n);
    end

    for j=n-1:-1:1
        coef(j, 3) = z(j)-m(j)*coef(j+1, 3);
        coef(j, 2) = (coef(j+1, 1) - coef(j, 1))/h(j) - h(j)*(coef(j+1,3)+2*coef(j,3))/3;
        coef(j, 4) = (coef(j+1, 3)-coef(j, 3))/(3*h(j));
    end

    coef = coef(1:end-1, :);
end