function res = Hermite(x_nodes, y_nodes, y_deriv_nodes, eval)
    n = size(x_nodes, 2);

    z = zeros(2 * n,1);
    z(1:2:end, 1) = x_nodes';
    z(2:2:end, 1) = x_nodes';

    dif_div = zeros(2 * n);
    dif_div(1:2:end, 1) = y_nodes';
    dif_div(2:2:end, 1) = y_nodes';
    dif_div(1:2:end, 2) = y_deriv_nodes';
    dif_div(2:2:end-1, 2) = ((dif_div(3:2:end, 1) - dif_div(2:2:end-1, 1)) ./ (z(3:2:end, 1) - z(2:2:end-1, 1)));

    for j = 3:2*n
        for i = 1:2*n-j+1
            dif_div(i, j) = (dif_div(i + 1, j - 1) - dif_div(i, j-1)) / (z(i+j-1) - z(i));
        end
    end

%     disp([z dif_div]);

    m = length(eval);
    res = zeros(1, m);
    for k=1:m
        p = dif_div(1, 1);
        term = eval(k) - z(1);
        for i=2:2*n
            p = p + term*dif_div(1, i);
            term = term * (eval(k) - z(i));
        end
        res(k) = p;
    end
end