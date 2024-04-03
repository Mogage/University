valueToCheck = pi / 3;
sin_approx = mySin(valueToCheck);
sin_approx2 = mySin2(valueToCheck);
cos_approx = myCos(valueToCheck);
cos_approx2 = myCos2(valueToCheck);
fprintf('The sinus and cosinus for value %f are: %.2f, %.2f | %.2f, %.2f.\n', valueToCheck, sin_approx, sin_approx2, cos_approx, cos_approx2);
