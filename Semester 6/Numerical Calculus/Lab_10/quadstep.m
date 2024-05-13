function res = quadstep(f, a, b, eps, fa, fc, fb)
    h = b - a;
    c = (a+b)/2;
    fd = f((a+c)/2);
    fe = f((c+b)/2);
    res1 = h/6*(fa+4*fc+fb);
    res2 = h/12*(fa+4*fb+2*fc+4*fe+fb);
    if (abs(res1 - res2) < eps)
        res = res2 + (res2-res1)/15;
    else
        resa = quadstep(f, a, c, eps, fa, fd, fc);
        resb = quadstep(f, c, b, eps, fc, fe, fb);
        res = resa+resb;
    end
end