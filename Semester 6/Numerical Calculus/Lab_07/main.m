nodes = [0 1 2];
y_values = exp(nodes);
y_derivatives = exp(nodes);
eval = -1 : 0.01 : 3;
res = Hermite(nodes, y_values, y_derivatives, eval);

plot(eval, exp(eval), 'b', 'LineWidth', 2);
hold on; 
plot(eval, res, 'r--', 'LineWidth', 2); 
xlabel('x'); 
ylabel('y'); 
title('Reprezentarea funcției f și a interpolării Hermite'); 
legend('f(x)', 'Interpolare Hermite', 'Location', 'best'); 
hold off; 


nodes = [-2 3];
y_values = [4 9]; 
y_derivatives = [-4 6];
eval = -3 : 0.01 : 3;
figure;
plot(eval, Hermite(nodes, y_values, y_derivatives, eval), "blue");
