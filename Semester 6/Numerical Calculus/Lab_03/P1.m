wilson = [10,7,8,7;
          7,5,6,5;
          8,6,10,9;
          7,5,9,10];

result = [32, 23, 33, 31]';
solution = wilson \ result;

% a
perturbedResult = [32.1,22.9,33.1,30.9]';
perturbedSolution = wilson \ perturbedResult;

entryError = norm(perturbedResult - result) / norm(result)
exitError = norm(perturbedSolution - solution) / norm(solution) 
fprintf ("Eroarea pentru membrul drept: %f\n", exitError / entryError);

%b)

perturbedMatrix = [10,7,8.1,7.2;
                   7.08,5.04,6,5;
                   8,5.98,9.89,9;
                   6.99,4.99,9,9.98];

perturbedSolution = perturbedMatrix \ result;
entryError = norm(wilson - perturbedMatrix) / norm(wilson)
exitError = norm(perturbedSolution - solution) / norm(solution)
fprintf("Eroarea pentru membrul stang: %f\n", entryError / exitError);