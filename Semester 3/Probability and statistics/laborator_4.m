pkg load statistics;

function problem1
  ham = fileread('keywords_ham.txt');
  hWords = strsplit(ham, ' ');
  u_hWords = unique(hWords);
  u_hWords = u_hWords(2:length(u_hWords));
  hFreq = [];
  Lh1 = length(u_hWords);
  Lh2 = length(hWords);
  for i = 1 : Lh1
    counter = 0;
    for j = 1 : Lh2
      counter = counter + strcmp(u_hWords(i), hWords(j));
    endfor
    hFreq = [hFreq, counter/Lh2];
  endfor


  email = fileread('email2.txt');
  eWords = strsplit(email, ' ');
  u_eWords = unique(eWords);
  u_eWords = u_eWords(2:length(u_eWords));
  eFreq = [];
  Le1 = length(u_eWords);
  Le2 = length(eWords);
  for i = 1 : Lh1
    counter = 0;
    for j = 1 : Le1
      counter = counter + strcmp(u_hWords(i), u_eWords(j));
    endfor
    if counter == 0
      hFreq(i) = 1 - hFreq(i);
    endif
  endfor

  spam = fileread('keywords_spam.txt');
  sWords = strsplit(spam, ' ');
  u_sWords = unique(sWords);
  u_sWords = u_sWords(2:length(u_sWords));
  sFreq = [];
  Ls1 = length(u_sWords);
  Ls2 = length(sWords);
  for i = 1 : Ls1
    counter = 0;
    for j = 1 : Ls2
      counter = counter + strcmp(u_sWords(i), sWords(j));
    endfor
    sFreq = [sFreq, counter/Ls2];
  endfor

  for i = 1 : Ls1
    counter = 0;
    for j = 1 : Le1
      counter = counter + strcmp(u_sWords(i), u_eWords(j));
    endfor
    if counter == 0
      sFreq(i) = 1 - sFreq(i);
    endif
  endfor

  probCHam = Lh2 / (Ls2 + Lh2);
  probCSpam = Ls2 / (Ls2 + Lh2);

  probHam = prod(hFreq) * probCHam
  probSpam = prod(sFreq) * probCSpam

  if probHam > probSpam
    disp("Email este ham.\n");
  else
    disp("Email este spam.\n");
  endif
end




function retval = f(p, k)
  X = binornd(1, p, 1, k);
  count = 0;
  vec = [];
  for i = 1 : k
    if X(i) == 1
      count += 1;
    else
      count -= 1;
    endif
    vec = [vec, count];
  endfor
  retval = count;
end

function problem2
  clf;  grid on;  hold on;
  vec = [f(0.5, 100)];
  k = 100;
  p = 0.5;
  for i = 1 : 100
    vec = [vec, f(p, k)];
  endfor
  lastFreq = hist(vec, -k : k);
  bar(-k : k,lastFreq/k,'hist','FaceColor','b');
end

function problem3
  m = 1000;
  p = sum(hygepdf(3:6, 49, 6, 6));
end

%problem1;
problem2;
