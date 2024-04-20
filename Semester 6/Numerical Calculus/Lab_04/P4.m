A = [1,2,1;
    2,5,3;
    1,3,3];
B = [4;10;7];
cholesky(A, B)

[A,B] = generare_matrici(10);
fprintf("Descompunere Cholesky: \n");
disp(cholesky(A, B));