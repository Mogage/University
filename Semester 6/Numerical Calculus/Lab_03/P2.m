%tk=-1+2*k/n
nValues = 10:15; 
conditionariA = zeros(1, length(nValues));

for i = 1:length(nValues)
    n = nValues(i);
    k = linspace(-1,1,n);
    t = -1+(2.*k)./n;
    V = vander(t);
    conditionariA(i) = cond(V, "inf");
end
conditionariA

%tk=1/k
conditionariB = zeros(1, length(nValues));
for i = 1:length(nValues)
    n = nValues(i);
    k = 1:n;
    o = ones(1,n);
    t = o./k;
    V = vander(t);
    conditionariB(i) = cond(V, "inf");
end
conditionariB

figure(1)
plot(10:15, log10(conditionariA), 'b*', 10:15, log10(conditionariB), 'r*')
title('Conditionarea matricei Vandermonde (tk = -1 + k^2/n) - blue si (tk = 1/k) - red')
xlabel('n')
ylabel('log10(cond(V))')