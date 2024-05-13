nodes = [0 3 5 8 13];
y_values = [0 225 383 623 993];
y_derivatives = [75 77 80 74 72];
eval = 10;

dist = Hermite(nodes, y_values, y_derivatives, eval);
disp([dist dist./eval]);