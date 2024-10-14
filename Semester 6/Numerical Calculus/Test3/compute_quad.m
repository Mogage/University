function [n, result] = compute_quad(func, eps, weight_type, a, b)
    n = 1;
    prev_result = -1;
    while true
        if nargin == 5
            [nodes, weights] = GaussQuadrature(n, weight_type, a, b);
        elseif nargin == 4
            [nodes, weights] = GaussQuadrature(n, weight_type, a);
        else
            [nodes, weights] = GaussQuadrature(n, weight_type);
        end
        result = weights * func(nodes);
        if abs(result - prev_result) < eps
            break;
        end
        prev_result = result;
        n = n + 1;
    end
end
