% function [A,B] = matDiag(n)
%     max_element = 50;
%     A = randi([-max_element,max_element],n);
%     out = sum(abs(A),2) + randi(max_element, n,1);
%     for i=1:n
%         A(i,i) = out(i);
%     end
%     B = A\[1:n]';
% end

function [A, B] = matDiag(n)
    A = rand(n);
    A = A +eye(n)*n;
    sol = zeros(n, 1);
    for i=1:n
        sol(i,1)=i;
    end
    B = A * sol;
end