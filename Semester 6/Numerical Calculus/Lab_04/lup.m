function result = lup(leftMatrix, rightMatrix)
    noRows = size(leftMatrix, 1);
    P = eye(noRows);

    for i=1:noRows-1
        [~, pivotRow] = max(abs(leftMatrix(i:noRows,i)));
        pivotRow = pivotRow + i - 1;

        if pivotRow ~= i
            P([i, pivotRow], :) = P([pivotRow, i], :);
            leftMatrix([i, pivotRow], :) = leftMatrix([pivotRow, i], :);
        end

        row = i + 1 : noRows;
        leftMatrix(row,i) = leftMatrix(row,i)/leftMatrix(i,i);
        leftMatrix(row,row) = leftMatrix(row,row) - leftMatrix(row,i)*leftMatrix(i,row);
    end

    L = tril(leftMatrix, -1) + eye(noRows);
    U = triu(leftMatrix);

    y = forwardSubstitution(L, P*rightMatrix);
    result = backwardSubstitution(U, y); 
end