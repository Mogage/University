function L = Lagrange(x_nodes, y_nodes, x)
    n = length(x_nodes);
    L = zeros(size(x));
    
    for k = 1:n
        l_i = ones(size(x));
        for j = 1:n
            if j ~= k
                l_i = l_i .* (x - x_nodes(j)) / (x_nodes(k) - x_nodes(j));
            end
        end
        L = L + y_nodes(k) * l_i;
    end
end