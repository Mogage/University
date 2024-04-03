function cos_approx = myCos(x)
    [x, ~, cosSign] = firstQuadrant(x);
    cos_approx = 0;
    seriesTerm = 1; % first term of the series
    n = 1;

    while cos_approx + seriesTerm ~= cos_approx
        cos_approx = cos_approx + seriesTerm;
        seriesTerm = (-1)^n * (x^(2*n) / factorial(2*n));
        n = n + 1;
    end

    if cos_approx >= 10^-10
        cos_approx = cosSign * cos_approx;
    end
end