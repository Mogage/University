function sin_approx = mySin2(x)
    [x, sinSign, ~] = firstQuadrant(x);
    sin_approx = 0;
    seriesTerm = x; % first term of the series
    n = 1;

    while sin_approx + seriesTerm ~= sin_approx
        sin_approx = sin_approx + seriesTerm;
        seriesTerm = x ^ 2 / ((n + 1) * (n + 2)) * -seriesTerm;
        n = n + 2;
    end
    
    sin_approx = sinSign * sin_approx;
end