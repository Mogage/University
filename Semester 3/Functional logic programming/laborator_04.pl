/*
                         l1; n=1, l1=sum
subListSum(l1..ln,sum) = subListSum(l2..ln, sum); n > 0
                         l1 (+) subListSum(l2..ln, sum-l1); sum - l1 > 0
subListSum(L:List, Sum:Integer, Res:List)
L -> Lista in care se cauta subliste a caror suma de elemente sa fie egala cu NumbersSum
Sum -> Suma cu care sa fie egala elementele sublistelor
Res -> O sublista a caror suma de elemente este egala cu Sum
(i,i,o) -> nedeterminist
*/
subListSum([Head | _], Head, [Head]).
subListSum([_ | Tail], Sum, Res) :- 
    subListSum(Tail, Sum, Res).
subListSum([Head | Tail], Sum, [Head | Res]) :- 
    Sum1 is Sum - Head,
    Sum1 > 0,
    subListSum(Tail, Sum1, Res).

/*
allSums(NumbersList:List, NumbersSum:Integer, Res:List)
NumbersList -> Lista in care se cauta subliste a caror suma de elemente sa fie egala cu NumbersSum
NumbersSum -> Suma cu care sa fie egala elementele sublistelor
Res -> Lista cu toate sublistele din NumbersList a caror suma este egala cu NumbersSum
(i,i,o) -> determinist
*/
allSums(NumbersList, NumbersSum, Res) :-
    findall(OutputList, subListSum(NumbersList, NumbersSum, OutputList), Res),
    print(Res).
    