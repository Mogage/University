function result = cholesky(leftMatrix, rightMatrix, solve)
    noRows = size(leftMatrix, 1);
    R = leftMatrix;
    for i=1:noRows
        for j=i+1:noRows
            R(j, j:noRows) = R(j, j:noRows) - R(i,j:noRows) * conj(R(i,j)) / R(i, i);
        end
        R(i, i:noRows) = R(i, i:noRows) / sqrt(R(i, i));
    end
    R = triu(R);
    
    if solve == false
        result = R;
        return 
    end

    y = forwardSubstitution((R.'), rightMatrix);
    result = backwardSubstitution(R, y); 
%   metoda de jos a fost folosita in laborator, dar la recomandarea 
%   dumneavoastra, am schimbat cu functiile care calculeaza manual
%     y = (R.')\rightMatrix;
%     result = R\y;
end