function L = LagrangeBari(x_nodes, y_nodes, x)
    w = zeros(size(x_nodes));
    n = length(x_nodes);
    
    for j = 1:n
        w(j) = 1 / prod(x_nodes(j) - x_nodes(1:j-1)) / prod(x_nodes(j) - x_nodes(j+1:end)); 
    end

    L = zeros(size(x));
    for j = 1:length(x)
        numerator = 0;
        denominator = 0;
        for k = 1:n
            if x(j) == x_nodes(k)
                L(j) = y_nodes(k);
                break;
            else
                weight = w(k) / (x(j) - x_nodes(k));
                numerator = numerator + weight * y_nodes(k);
                denominator = denominator + weight;
            end
        end
        L(j) = numerator / denominator;
    end
end