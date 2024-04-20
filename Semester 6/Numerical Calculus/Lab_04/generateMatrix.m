function [A,B] = generateMatrix(n)
    max = 50;
   
    while true
      A = ceil(unifrnd(-max, max,n));
      if det(A) ~= 0; break; end 
    end
    
    B = sum(A,2);
end