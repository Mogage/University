nodes = [0.3 0.32 0.35];
y_values = sin(nodes);
y_derivatives = cos(nodes);
eval = 0.34;

pred = Hermite(nodes, y_values, y_derivatives, eval);
true = sin(eval);
disp([pred true]);
disp(true-pred);

nodes = [0.3 0.32 0.33 0.35];
y_values = sin(nodes);
y_derivatives = cos(nodes);
eval = 0.34;

pred = Hermite(nodes, y_values, y_derivatives, eval);
true = sin(eval);
disp([pred true]);
disp(true-pred);