perms('word')

nchoosek([2,3,5,7], 3)

function laborator_1_function(vector, k)
  combinari = nchoosek(vector, k);
  [nrL, nrC] = size(combinari);
  disp('aranjamente: ')
  for i = 1 : nrL
    disp(perms(combinari([i],:)));
  endfor
end

laborator_1_function([2,3,5,7], 3)
