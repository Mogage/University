syms x;
valueToCheck = 7 * pi / 2;
[~, sinSign, cosSign] = firstQuadrant(valueToCheck);
m = 4;
k = 2;
f = sin(x);
resultSin = sinSign * myPade(f, m, k, valueToCheck);

g = cos(x);
resultCos = cosSign * myPade(g, m, k, valueToCheck);

fprintf('The sinus and cosinus for value %.2f are: %.2f | %.2f.\n', valueToCheck, resultSin, resultCos);