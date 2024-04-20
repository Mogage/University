[A,B] = generare_matrici(10);

fprintf("Eliminare gausiana: \n");
disp(gauss(A,B));
fprintf("Descompunere LUP: \n");
disp(lup(A, B));