/*
                            false; Number < Count, Number > 1
divisors0(Number, Count) =  Count (+) divizori(Number, Count+1); Number%Count=0, Number < Count, Number > 1
                            divizori(Number, Count+1), Number < Count, Number > 1

divisors0(Number^i, Count^i, Res^o)
Number -> Numarul pentru care se cauta divizorii
Count -> Numarul care se verifica daca este divizor al lui Number
Res -> Lista cu toti divizorii a lui Number
*/
divisors0(Number, Number, []).
divisors0(Number, _, []) :- 
    1 is Number, !.
divisors0(Number, Count, [Count | Res]) :- 
    Count < Number, 
    0 is Number mod Count,
    Count1 is Count+1, 
    divisors0(Number, Count1, Res).
divisors0(Number, Count, Res) :- 
    Count < Number, 
    Count1 is Count+1, 
    divisors0(Number, Count1, Res).

/*
divisors(Number) = Res; divisors0(Number, 2, Res)

divisors(Number^i, Res^o)
Number -> Numarul pentru care se cauta divizorii
Res -> Lista cu toti divizorii a lui Number
*/
divisors(Number, Res) :- 
    divisors0(Number, 2, Res).

/*
List1 = l1l2..ln
List2 = l'1l'2...l'm
addSubList(List1, List2) =  List2; n = 0
                            inceput(l1) (+) addSubList(l2l3...ln); altfel

addSubList(List1^i, List2^i, Res^o)
List1 -> Prima Lista de reunit
List2 -> A doua Lista de reunit
Res -> Lista cu Lista2 adaugata dupa Lista1
*/
addSubList([], List, List).
addSubList([Head | Tail1], List2, [Head | Tail2]) :- 
    addSubList(Tail1, List2, Tail2).


/*
                    []; n = 0
addDivisors(List) = l1 (+) addDivisors(l2l3..ln); prim(l1)
                    l1 (+) addSublist(divisors(l1), addDivisors(l2l3..ln)); altfel

addDivisors(List^i, Res^o)
List -> Lista cu numere naturale
Res -> Lista List la care se adauga divizorii proprii a numerelor
*/
addDivisors([], []).
addDivisors([Head | Tail], Res) :-
    divisors(Head, DivisorsList), 
    addDivisors(Tail, Res2),
    addSubList(DivisorsList, Res2, Res3),
    Res = [Head | Res3], !.

/*
                        []; n = 0  
secondProblem(List) =   l1 (+) secondProblem(l2l3..ln); l1 este numar
                        addDivisors(l1) (+) secondProblem(l2l3..ln); l1 este lista

secondProblem(List^i, Res^o)
List -> Lista cu elemente Liste si Numere
Res -> Lista List la care se adauga divizorii proprii a numerelor din sub liste
*/
secondProblem([], []).
secondProblem([Head | Tail], Res) :- 
    is_list(Head), 
    addDivisors(Head, Res2),
    secondProblem(Tail, Res3), 
    Res = [Res2 | Res3], !.
secondProblem([X | Tail], [X | Res]) :- 
    secondProblem(Tail, Res).
