function result = gauss(leftMatrix, rightMatrix)
    noRows = size(leftMatrix, 1);
    result = zeros(noRows, 1);
    extendedMatrix = [leftMatrix, rightMatrix];

    for i=1:noRows-1
        [pivotValue, pivotRow] = max(abs(leftMatrix(i:noRows,i)));
        pivotRow = pivotRow + i - 1;
    
        if pivotValue == 0
            error("Solution is not unique");
        end
        if pivotRow ~= i
            extendedMatrix([i, pivotRow], :) = extendedMatrix([pivotRow, i], :);
        end
        for j=i+1:noRows
            m = extendedMatrix(j,i) / extendedMatrix(i,i);
            extendedMatrix(j, :) = extendedMatrix(j, :) - m * extendedMatrix(i, :);
        end
    end

    if extendedMatrix(noRows, noRows) == 0
        error("Solution is not unique");
    end

    result(noRows) = extendedMatrix(noRows, noRows+1)/extendedMatrix(noRows, noRows);
    for i=noRows-1:-1:1
        sum = 0;
        for j = i+1:noRows
          sum = sum + extendedMatrix(i, j) * result(j);
        end
        result(i) = (extendedMatrix(i, noRows+1) - sum) / extendedMatrix(i, i);
    end
end
