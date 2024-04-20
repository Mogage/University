function y = forwardSubstitution(L, rightMatrix)
  n = size(L, 1);
  y = zeros(n, 1);

  for i = 1:n
    sum = 0;
    for j = 1:i-1
      sum = sum + L(i, j) * y(j);
    end
    y(i) = (rightMatrix(i) - sum) / L(i, i);
  end
end