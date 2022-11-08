%(i,i,o)
interval(N,N,[N]).
interval(M,N,[M|Rez]) :- M < N, M2 is M + 1, interval(M2, N, Rez).

%(i, i)
member1(E,[E|_]) :- !.
member1(E,[H|L]) :- E\=H, member1(E,L), !.

%(i,i,o)
intersectie([],_,[]).
intersectie([X|L1],L2,[X|Res]) :- member1(X,L2), intersectie(L1, L2, Res), !.
intersectie([_|L1],L2,Res) :- intersectie(L1, L2, Res).
