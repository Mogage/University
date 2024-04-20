n = 10;
err = 1e-6;
main_diag = 5*ones(n,1);
upper_diag = -1*ones(n-1,1);
second_diag = -1*ones(n-3,1);

% first matrix
A1 = diag(main_diag) + diag(upper_diag,1) + diag(upper_diag,-1);
B1 = A1 * ones(n,1);
% second matrix
A2 = diag(main_diag) + diag(upper_diag,1) + diag(upper_diag,-1) + diag(second_diag, 3) + diag(second_diag, -3);
B2 = A2 * ones(n,1);
% third matrix
[A3, B3] = matDiag(25);

% testMethods(A1, B1, err);
% testMethods(A2, B2, err);
testMethods(A3, B3, err);
