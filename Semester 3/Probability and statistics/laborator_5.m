% Generare de valori pentru o variabila discreta

% v = [v1, v2, v3];

% u = rand; 0 <= u <= 1

% cumsum(v) = [v1, v1 + v2, v1 + v2 + v3] ==> [0, v1], [v1, v1 + v2], [v1 + v2, v1 + v2 + v3]

% v1 + v2 <= u <= v1 + v2 + v3 ==> generam v3

function x = rndvardisc(v, p, n)

  q = cumsum(p);
  x = zeros(1, n);

  for i = 1 : n
    u = rand;
    j = 1;

    while u > q(j)
      j++;
    endwhile

    x(i) = v(j);
  endfor
endfunction

function pb1(v = [0, 1, 2, 3], p = [0.46, 0.4, 0.1, 0.04], n = 1000)

  clf; grid on; hold on;

  x = rndvardisc(v, p, n);
  y = randsample(v, n, replacement = true, p)

  rel_frq_x = hist(x, v) / n;
  bar(v, rel_frq_x, 'hist', 'FaceColor', 'b');

  rel_frq_y = hist(y, v) / n;
  bar(v, rel_frq_y, 'FaceColor', 'y');

  set(findobj('type', 'patch', 'facealpha', 0.7);

endfunction


% Generare de valori folosind metoda inversei


function x = rndexp(lambda, n)

  x = - log(1 - rand(1, n)) * lambda;

endfunction

function pb2(lambda, n)

  x = rndexp(lambda, n)
  y = exprnd(lambda, 1, n)

  [mean(x), mean(y)]
  (std(x), std(y)]

endfunction

function [x, y] = boxmuller(n)

  u = rand(2, n);

  r = sqrt( -2 * log(u(1, :)));
  v = 2 * pi * u(2, :);

  x = r .* cos(v);
  y = r .* sin(v);

endfunction

function pb3(n = 500)

  clf;

  t = linspace(0, 2 * pi, 360);
  polar(t, 4 * ones(1, 360), 'w');
  hold on;

  [x, y] = boxmuller(n);
  z =  normrnd(0, 1, 2, n);

  plot(x, y, 'r*');
  plot(z(1, :), z(2, :), 'c*');
  polar(t, 0.5 * ones(1, 360), 'b');

  bulls_eye_bm = mean(x .^ 2 + y .^ 2 < 0.25);
  bulls_eye_z = mean(z(1, :) .^ 2 + z(2, :) .^ < 0.25);
  1 - exp(-1 / 8);

  % probabilitate teoretica
  % P(sqrt(x ^ 2 + y ^ 2) < 0.5) = P (sqrt(-2 * log(u1)) < 0.5) =
  % = P( -2 * log(u1) < 0.25) = P(log(u1) > -1 / 8) = P(u1 > exp(-1 / 8)

endfunction


















