function res = adquad(f, a, b, eps)
    c = (a+b)/2;
    res = quadstep(f, a, b, eps, f(a), f(c), f(b));
end