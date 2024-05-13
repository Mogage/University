function result = backwardSubstitution(U, y)
    n = size(U, 1);
    result = zeros(n, 1);

    for i=n:-1:1
        sum = 0;
        for j=i+1:n
            sum = sum + U(i,j)*result(j);
        end
        result(i) = (y(i) - sum) / U(i, i);
    end
end