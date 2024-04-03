function cond_number = condPoly(p, r)
    if nargin < 2
        r = roots(p);
    end
    n=length(p)-1;
    deriv=[n:-1:1].*p(1:end-1);
    valoareDeriv=polyval(deriv,r);
    valoarePoly=polyval(abs(p(2:end)), abs(r));
    cond_number=valoarePoly./(abs(r.*valoareDeriv));
end
