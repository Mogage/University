function testMethods(A, B, err)
    disp(Jacobi(A, B, err));
    disp(GaussSeidel(A, B, err));
    disp(SOR(A, B, computeOmega(A), err));
end