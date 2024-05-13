function res = evalSpline(nodes, coef, eval)
    n = length(eval);
    res = zeros(1, n);
    for i=1:n
        idx = find(eval(i) >= nodes(1:end-1) & eval(i) <= nodes(2:end), 1, 'first');
        if (isempty(idx))
            if (eval(i) > nodes(end))
                idx = length(nodes) - 1;
            else
                idx = 1;
            end
        end
        term = eval(i) - nodes(idx);
        res(i) = coef(idx, 1) + term * coef(idx, 2) + term ^ 2 * coef(idx, 3) + term ^ 3 * coef(idx, 4);
    end
end
