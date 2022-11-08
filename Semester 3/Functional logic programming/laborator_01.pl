goLast([Head | Tail]) :- goLast(Head, Tail).
goLast(Numar, [Head | Tail]) :- goLast(Head, Tail).
goLast(Numar, []) :- print(Numar).

remover(_, _, [], [], COUNT).
remover(R, N, [R|T], T2, COUNT) :- remover( R, N, T, T2, 0).
remover(R, N, [H|T], [H|T2], COUNT) :- N=COUNT, H \= R, remover( R, N, T, T2, COUNT+1).
remover(R, N, [H|T], [H|T2]) :- remover(R, N, [H|T], T2, 0).