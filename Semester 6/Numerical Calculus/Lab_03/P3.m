radacini = [1:20];
polinom = poly(radacini);
fprintf('Numar conditionare pentru (x-1)(x-2)...(x-20)')
condPoly(polinom,radacini)
computeCond(polinom, radacini, 1);

numb = [1:20];
polinom = 2.^-numb;
radacini = roots(polinom);
fprintf('Numar conditionare pentru a_k=2^-k, k=1..20')
transpose(condPoly(polinom,radacini))
computeCond(polinom, radacini, 2);
