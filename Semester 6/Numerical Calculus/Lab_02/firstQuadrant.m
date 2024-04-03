function [f, semn_sin, semn_cos] = firstQuadrant(x)
    x = mod(x, 2*pi);
    
    if (x >= 0 && x <= pi/2)
        semn_sin = 1;
        semn_cos = 1;
        f = x; 
    elseif (x > pi/2 && x <= pi)
        semn_sin = 1;
        semn_cos = -1;
        f = pi - x;
    elseif (x > pi && x < 3*pi/2)
        semn_sin = -1;
        semn_cos = -1;
        f = x - pi;
    else
        semn_sin = -1;
        semn_cos = 1;
        f = 2*pi - x;
    end
end