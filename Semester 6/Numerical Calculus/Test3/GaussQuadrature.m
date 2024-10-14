function [nodes, weights] = GaussQuadrature(n, weight_type, a, b)
    
    if ~ismember(weight_type, {'legendre', 'chebyshev1', 'chebyshev2', 'laguerre', 'hermite', 'jacobi'})
        error('Invalid weight type. Choose from: legendre, chebyshev1, chebyshev2, laguerre, hermite, jacobi');
    end

    if strcmp(weight_type, 'legendre')
        beta = [2, (4 - (1:n - 1).^(-2)).^(-1)]; 
        alpha = zeros(n, 1);
    elseif strcmp(weight_type, 'chebyshev1')
        nodes = cos(pi*([1:n]'-0.5)/n);
        weights = pi / n * ones(1, n);
        return
    elseif strcmp(weight_type, 'chebyshev2')
        beta = [pi / 2, 1 / 4 * ones(1, n-1)];
        alpha = zeros(n, 1);
    elseif strcmp(weight_type, 'jacobi')
        alpha0 = (b-a) / (b+a+2);
        beta0 = 2 ^ (a+b+1) * gamma(a+1) * gamma(b+1) / gamma(a+b+2);
        if n == 1
            alpha = alpha0;
            beta = beta0;
        else 
            if a==b
                alpha=zeros(1,n);
            else
                alpha=[alpha0, (b^2-a^2)./(2*(1:n-1)+a+b)./(2*(1:n-1)+a+b+2)];
            end
            beta1 = 4 * (1+a) * (1+b) / ((2+a+b)^2 * (3+a+b));
            k = 2:n-1;
            beta = [beta0, beta1, (4*k.*(k+a).*(k+a+b).*(k+b))./((2*k+a+b-1).*(2*k+a+b).^2.*(2*k+a+b+1))];
        end
    elseif strcmp(weight_type, 'laguerre')
        beta = [gamma(1+a), (1:n-1).*((1:n-1)+a)];
        alpha = [a+1, 2*(1:n-1)+a+1];
    elseif strcmp(weight_type, 'hermite')
        beta = [sqrt(pi), 1:n-1 / 2];
        alpha = zeros(n, 1);
    end
    
    sqrt_beta = sqrt(beta(2:n));
    J = diag(alpha) + diag(sqrt_beta, -1) + diag(sqrt_beta, 1); 
    [V, D] = eig(J);                    
    nodes = diag(D);                   
    weights = beta(1) * V(1,:).^2;
end
