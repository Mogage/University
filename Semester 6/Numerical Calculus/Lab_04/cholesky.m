function result = cholesky(leftMatrix, rightMatrix)
    noRows = size(leftMatrix, 1);
    R = leftMatrix;
    for i=1:noRows
        for j=i+1:noRows
            R(j, j:noRows) = R(j, j:noRows) - R(i,j:noRows) * conj(R(i,j)) / R(i, i);
        end
        R(i, i:noRows) = R(i, i:noRows) / sqrt(R(i, i));
    end
    R = triu(R);

    y = (R.')\rightMatrix;
    result = R\y;
end