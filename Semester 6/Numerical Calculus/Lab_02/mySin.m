function sin_approx = mySin(x)
    [x, sinSign, ~] = firstQuadrant(x);
    sin_approx = 0;
    seriesTerm = x; % first term of the series
    n = 1;

    while sin_approx + seriesTerm ~= sin_approx
        sin_approx = sin_approx + seriesTerm;
        seriesTerm = (-1)^n * (x^(2*n+1) / factorial(2*n+1));
        n = n + 1;
    end
    
    sin_approx = sinSign * sin_approx;
end