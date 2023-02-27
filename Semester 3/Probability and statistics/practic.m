% zeros(nr_coloane, nr_linii) -> genereaza o matrice cu zerouri
% unifrnd(a, b, nr_coloane, nr_linii -> genereaza o matrice cu valori din [a,b]\
% randsample(matrice/vector, nr_elemente) -> extrage un numar de elemente din m/v
% mean(vector) -> calculeaza media elementelor din vector

pkg load statistics

function practic1
  suma = zeros(1, 1000);
  probabilitate = 0;
  for i = 1 : 1000
    vector = unifrnd(0, 10, 1, 5);
    x = randsample(vector, 2);
    suma(i) = x(1) + x(2);
    diferenta = abs(x(1) - x(2));
    if diferenta > 2
      probabilitate++;
    endif
  endfor

  medie = mean(suma);

  disp(medie);
  disp(probabilitate/1000);

  probabilitate_interval=unifcdf(8,0,10)-unifcdf(5,0,10)
  probabilitate_2_elemente=binopdf(2,5,probabilitate_interval)
  binornd(4,probabilitate_interval,1,10)
endfunction

function practic2
  % a)
  disp(7/10);
  % b)
  timp_server = [1 2 3 4];
  prob = [1/10 2/10 3/10 4/10];

  srv = randsample(timp_server, 10, true, prob);
  timpi = exprnd(srv);
  mean(timpi)
%c)
  favorabile=sum(timpi>=3);
  prob=favorabile/10

endfunction

function [p1, p2, p3] = aux_r41(vec)
  p1 = 0;
  p2 = 0;
  p3 = 0;
  if vec(1) == 'a' && vec(2) == 'r' && vec(3) == 'v'
    %disp("da");
    p1 = 1;
  endif
  if !any(vec == 'v')
    %disp("nu are verzi");
    p2 = 1;
  endif
  if sum(vec == 'v') == 1
    %disp("exact una verde");
    p3 = 1;
  endif
endfunction

function r41
  bile = "rrrrrrrrrraaavvvvv";

  probs_return = [0 0 0];
  probs_w8_return = [0 0 0];

  for i = 1 : 1000
    with_return = randsample(bile, 3, true);
    [p1, p2, p3] = aux_r41(with_return);
    probs_return += [p1, p2, p3];
    without_return = randsample(bile, 3);
    [p1, p2, p3] = aux_r41(without_return);
    probs_w8_return += [p1, p2, p3];
  endfor
  disp(probs_return ./ 1000);
  disp(probs_w8_return ./ 1000);

endfunction

function media(n=1000)
  pasi=ceil(normrnd(170,10,1,n));
   medie=mean(pasi)
  favorabile=0;
  for i=1:length(pasi)
    if pasi(i)>200
      favorabile=favorabile+1;
    endif
  endfor
  procentaj=favorabile/n*100
  favorabile
  max_pasi=max(pasi);
  min_pasi=min(pasi);
  posibile=min_pasi:max_pasi;
  frecv_absolute=hist(pasi,posibile); %frecvente absolute


  %history -c;
  hold on;
  x=[min_pasi:0.1:max_pasi];
  %graficul functiei de densitate

  bar(posibile,frecv_absolute/n,'hist','FaceColor','y');
  #plot(x,normpdf(x,170,10),'r*')
  figure;
  plot(x,normcdf(x,170,10),'b*') %functia de repartitie

endfunction

media(1000);













