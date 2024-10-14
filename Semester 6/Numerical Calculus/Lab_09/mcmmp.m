function res = mcmmp(nodes, values, base, eval)
    X = base(nodes)';
    A = X' * X;   
    b = X' * values';
    a = (A \ b)';
    res = sum(a .* base(eval)', 2)';
end

% function res = mcmmp(nodes, values, n, eval)
%     base = ones(length(nodes), 2*n+1);
%     for i = 1:2*n
%         base(:, i+1) = nodes.^i;
%     end
%     rightMatrix = zeros(n+1, 1);
%     leftMatrix = zeros(n+1);
%     for i = 1:n+1
%         for j = 1:n+1
%             leftMatrix(i, j) = sum(base(:, j+i-1));
%         end
%         rightMatrix(i) = sum(values' .* base(:, i));
%     end
%     a = linsolve(leftMatrix, rightMatrix)';
%     evalBase = ones(length(eval), n+1);
%     for i = 1:n
%         evalBase(:, i+1) = eval.^i;
%     end
%     res = zeros(1, length(eval));
%     for i = 1:length(eval)
%         res(i) = sum(a .* evalBase(i, :));
%     end
% end