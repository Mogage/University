function res = adaptquad(f, a, b, eps, met)
    if (abs(met(f, a, b, 5) - met(f, a, b, 10)) < eps) 
        res = met(f, a, b, 10);
    else
        res = adaptquad(f, a, (a+b)/2, eps, met) + adaptquad(f, (a+b)/2, b, eps, met);
    end
end