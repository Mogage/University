function plotFundamentals(nodes)
    x = linspace(min(nodes), max(nodes), 1000); 
    n = length(nodes);
    L = zeros(n, length(x));

    for k = 1:n
        l_i = ones(size(x));
        for j = 1:n
            if j ~= k
                l_i = l_i .* (x - nodes(j)) / (nodes(k) - nodes(j));
            end
        end
        L(k,:) = l_i;
    end
    
    figure;
    for i = 1:n
        plot(x, L(i, :), 'DisplayName', ['l_', num2str(i)], 'LineWidth', 1.5);
        hold on;
    end
    
    title('Fundamental Polynomials of Lagrange Interpolation');
    xlabel('x');
    ylabel('l_i(x)');
    legend('show');
    hold off;
end
