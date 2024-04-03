function computeCond(polinom, radacini, figureNumber)
    figure(figureNumber)
    h = plot(radacini, zeros(1, length(radacini)), 'k.');
    
    set(h,'Markersize', 21);
    hold on
    for i=1:20
        polinomNorm=polinom+normrnd(0,1e-10,1,length(polinom));
        rootsNormA=roots(polinomNorm);
    
        h2 = plot(rootsNormA,'b*');
     
        hold on
        set(h2,'Markersize',10);
    
        polinomUniform = polinom+unifrnd(0,1e-10,1,length(polinom));
        rootsUniformA = roots(polinomUniform);
        
        h3=plot(rootsUniformA,'r*');
        hold on
        set(h3,'Markersize',6)
    
    end
    legend('radacini', 'norm distrib', 'uniforma');
    hold off
end