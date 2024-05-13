function ex2(n, solve)
    [A, B] = matrixGeneration(n);

    if solve
        disp(['LUP pentru ', num2str(n)]);
    else
        disp(['Descompunere LUP pe A pentru ', num2str(n)]);
    end
    disp(lup(A, B, solve));

    if solve
        disp(['Cholesky pentru ', num2str(n)]);
    else
        disp(['Descompunere Cholesky pe pentru ', num2str(n)]);
    end
    disp(cholesky(A, B, solve));
end