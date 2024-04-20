function omega = computeOmega(A)
    D = diag(diag(A));
    L = tril(A, -1);
    U = triu(A, 1);
    J = D\(L+U);
    e = eig(J);
    p = max(abs(e));
    omega = 2 / (1 + sqrt(1 - p^2));
end