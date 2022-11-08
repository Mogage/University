pkg load statistics;

function problem1

  urna = 'rrrrraaavv';
  caseA = 0;
  caseB = 0;
  caseAB = 0;

  for i = 1 : 5000
    extragere = randsample(urna, 3);

    if any(extragere == 'r')
      caseA++;
    endif
    if all(extragere == 'a')
      caseB++;
    endif
    if all(extragere == 'r')
      caseB++;
      caseAB++;
    endif
  endfor

  disp(caseA / 5000);
  disp(caseB / 5000);
  disp(caseAB / 5000);
  disp(caseAB / caseA);
end

function problem2
  clf;  grid on;  hold on;
  p = 1/3;
  n = 5;
  m = 2000;

  x = binornd(n,p,1,m);
  N = hist(x,0:n);
  bar(0:n,N/m,'hist','FaceColor','b');
  bar(0:n,binopdf(0:n,n,p),'FaceColor','y');
  legend('probabilitatile estimate','probabilitatile teroretice');
  set(findobj('type','patch'),'facealpha',0.7);
  xlim([-1 n+1]);
end

function problem3
  zaruri = randi(6, 4, 1000);
  suma = sum(zaruri);

  pos = 4:24;
  h = hist(suma, pos);
  A = [pos;h];

  clf;grid on;hold on;
  xticks(pos);
  ylim([3,25]);
  yticks(0:0.01:0.14);
  ylim([0 0.14]);
  bar(pos,h/1000,'hist','FaceColor','b');

  disp(A);


  teoretice = []

  for i1
    for i2
      for i3
        for i4
          teo = [teo, i1+i2+i3+i4];
        endfor
      endfor
    endfor
  endfor

end

...problem1;
...problem2;
problem3;




















