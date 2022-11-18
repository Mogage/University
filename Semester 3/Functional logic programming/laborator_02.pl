/* 
interval(M, N) = [N] ; M=N
                 M (+) interval(M+1, N); M<N
instersectie(M: Integer, N: Integer, Rez: List)
M -> capat inferior al intervalului
N -> capat superior al intervalului
Rez -> lista cu numerele intregi din [M,N]
(i,i,o) -> determinist
*/
interval(N,N,[N]).
interval(M,N,[M|Rez]) :- M < N, M2 is M + 1, interval(M2, N, Rez).

/*
                    false; n = 0
member(e, L1..Ln) = true; L1 = e
                    member(e, L2..Ln); altfel
member(E:Integer, L:List)
E -> Element de verificat daca este in lista
L -> Lista in care se cauta
(i, i) -> determinist
*/
member1(E,[E|_]) :- !.
member1(E,[H|L]) :- E\=H, member1(E,L), !.

/* 
                                  []; n = 0
intersectie(L11..L1n, L21..L2n) = L11 (+) intersectie(L12..L1n, L21..L2n); member(L11, L2)
                                  intersectie(L12..L1n, L21..L2n); altfel
instersectie(L1: List, L2: List, Rez: List)
L1 -> prima multime
L2 -> a doua multime
Rez -> intersectia multimilor
(i,i,o) -> determinist
*/
intersectie([],_,[]).
intersectie([X|L1],L2,[X|Res]) :- member1(X,L2), intersectie(L1, L2, Res), !.
intersectie([_|L1],L2,Res) :- intersectie(L1, L2, Res).
