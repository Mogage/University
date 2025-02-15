function x = Jacobi(A,B,err)
    [m,n] = size(A);
    if (m ~= n) || (n ~= length(B))
        error('Invalid dimensions');
    end

    if det(A) == 0
        error('Matrix A is singular');
    end
    
    x = zeros(n,1);
    D = diag(A);
    L = tril(A, -1);
    U = triu(A, 1);
    D_inv = diag(1./D);
    alfa = norm(diag(D)\(L+U), inf);
    err = (1-alfa)/alfa*err;
    
    while true
        x_new = D_inv * (B - L*x - U*x);
    
        if norm(x_new - x, inf) < err
            x = x_new;
            break;
        end
   
        x = x_new;
    end
end 
