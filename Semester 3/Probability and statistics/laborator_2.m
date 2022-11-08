pkg load statistics

function problem1

  cazPos = 1000;
  cazFav = 0;

  for i = 1 : cazPos

    if length(unique(randi(365, 1, 23))) < 23
      cazFav++;
    endif

  endfor

  disp(cazFav / cazPos);

end

function problem2
  clf;
  hold on;
  axis square;
  rectangle("Position", [0, 0, 1, 1]);
  N = 500;
  frecCircle = 0;
  frecDistance = 0;
  frecTr = 0;
  for i = 1 : N
    x = rand;
    y = rand;
    centerDistance = pdist([x,y;0.5,0.5]);
    leftDownPoint = pdist([x,y;0,0]);
    rightDownPoint = pdist([x,y;1,0]);
    leftUpPoint = pdist([x,y;0,1]);
    rightUpPoint = pdist([x,y;1,1]);
    minimPoint = min([leftDownPoint, rightDownPoint, leftUpPoint, rightUpPoint]);
    trObtuz = 0;
    plot(x,y, 'dm', 'MarkerSize', 1, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
    if (x - 1/2)^2 + (y - 1/2)^2 <= 1/4
      plot(x,y, 'dm', 'MarkerSize', 1.5, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
      frecCircle++;
    endif
    if centerDistance <= minimPoint
        plot(x,y, 'dm', 'MarkerSize', 1.5, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g');
        frecDistance++;
    endif
    if leftDownPoint^2 + rightDownPoint^2 < 1
      trObtuz++;
    endif
    if leftDownPoint^2 + leftUpPoint^2 < 1
      trObtuz++;
    endif
    if leftUpPoint^2 + rightUpPoint^2 < 1
      trObtuz++;
    endif
    if rightUpPoint^2 + rightDownPoint^2 < 1
      trObtuz++;
    endif
    if 2 == trObtuz
      plot(x,y, 'dm', 'MarkerSize', 1.5, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'y');
      frecTr++;
    endif
  endfor
  disp(frecCircle/N);
  disp(frecDistance/N);
  disp(frecTr/N);
end

problem1;
problem2;
