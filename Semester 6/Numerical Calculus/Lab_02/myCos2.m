function cos_approx = myCos2(x)
    [x, ~, cosSign] = firstQuadrant(x);
    cos_approx = 0;
    seriesTerm = 1; % first term of the series
    n = 1;

    while cos_approx + seriesTerm ~= cos_approx
        cos_approx = cos_approx + seriesTerm;
        seriesTerm = x ^ 2 / (n * (n + 1)) * -seriesTerm;
        n = n + 2;
    end

    if cos_approx >= 10^-10
        cos_approx = cosSign * cos_approx;
    end
end