function [A, B] = matrixGeneration(n)
    main_diag = 3*ones(n,1);
    second_diag = 0.5*ones(n, 1);
    ofsset_diag = -1*ones(n - 1,1);
    second_diag(n / 2) = 0;
    second_diag(n / 2 + 1) = 0;
    
    A = diag(main_diag) + flip(diag(second_diag)) + diag(ofsset_diag, -1) + diag(ofsset_diag, 1);
    B = A * ones(n, 1);
end