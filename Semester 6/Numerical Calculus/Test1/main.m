% 1 - Complexitatea metodei lui Gauss pentru n=4
% Ne vom folosi de functia implementata in laborator pentru calcularea
% complexitatii. Facem abstractie de initializarea variabilelor result si
% extendedMatrix. Notatia ~= este folosita in sensul de aproximativ egal.
% function result = gauss(leftMatrix, rightMatrix)
%     noRows = size(leftMatrix, 1);
%     result = zeros(noRows, 1);
%     extendedMatrix = [leftMatrix, rightMatrix];
% 
%     for i=1:noRows-1
%         [pivotValue, pivotRow] = max(abs(leftMatrix(i:noRows,i)));
%         pivotRow = pivotRow + i - 1;
%     
%         if pivotValue == 0
%             error("Solution is not unique");
%         end
%         if pivotRow ~= i
%             extendedMatrix([i, pivotRow], :) = extendedMatrix([pivotRow, i], :);
%         end
%         for j=i+1:noRows
%             m = extendedMatrix(j,i) / extendedMatrix(i,i);
%             extendedMatrix(j, :) = extendedMatrix(j, :) - m * extendedMatrix(i, :);
%         end
%     end
% 
%     if extendedMatrix(noRows, noRows) == 0
%         error("Solution is not unique");
%     end
% 
%     result(noRows) = extendedMatrix(noRows, noRows+1)/extendedMatrix(noRows, noRows);
%     for i=noRows-1:-1:1
%         sum = 0;
%         for j = i+1:noRows
%           sum = sum + extendedMatrix(i, j) * result(j);
%         end
%         result(i) = (extendedMatrix(i, noRows+1) - sum) / extendedMatrix(i, i);
%     end
% end
% In cazul n(noRows) = 4, primul ciclu se executa de O(n-1)=O(3) ori, ca apoi
% gasirea pivotului sa fie facut in O(n-i) pasi, dar poate fi considerat ca
% O(n)=O(4). Daca pivotul este diferit de linia curenta, se mai adauga
% complexitatii parcurgerea matricei extinse, care se face in O(n+1)=O(5)
% pasi. Ulterior dupa schimbarea liniilor din matrice, intram iarasi
% intr-un ciclu de complexitate O(n-i)~=O(n)=O(4), in care valorile
% liniilor sunt actualizate, operatie care consta in O(n+1)=O(5) pasi. In
% final pentru calcularea rezultatului, se folosesc 2 cicluri imbricate
% care au o complexitate de O((n-1)*(n-i))~=O(n^2)=O(16).
% Dupa inmultirile si adunarile corespunzatoare, putem aproxima ca se
% efectueaza O(3*(4+5+4*5)+16)=O(103) pasi, in functie de alegerea
% pivotului putand sa fie efectuati mai putini.

% 2.a
ex2(4, false);
% 2.b
% ex2(100, true);
