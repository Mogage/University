function x = SOR(A,B,omega,err)
    [m,n] = size(A);
    if (m ~= n) || (n ~= length(B))
        error('Invalid dimensions');
    end

    if det(A) == 0
        error('Matrix A is singular');
    end
    if (omega<=0) || (omega>=2)
        error('Invalid relaxation parameter')
    end

    x = zeros(n,1);
    D = diag(A);
    L = tril(A, -1);
    U = triu(A, 1);
    D_inv = diag(1./D);
    T = (diag(D)-omega*L)\((1-omega)*diag(D)+omega*U);
    alfa = norm(T, inf);
    err = (1-alfa)/alfa*err;
    
    while true
        x_new = x;
        for i=1:n
            x_new(i) = (1-omega)*x(i) + omega * D_inv(i,i) * (B(i) - sum(L(i,:) * x_new) - sum(U(i,:) * x));
        end
    
        if norm(x_new - x, inf) < err
            x = x_new;
            break;
        end
   
        x = x_new;
    end
end