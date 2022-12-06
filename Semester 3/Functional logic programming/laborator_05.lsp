; Problema 1
; produsScalar(x1,y1,x2,y2) = x1*x2+y1*y2
; x1 int - coordonata pe axa OX a primului vector
; y1 int - coordonata pe axa OY a primului vector
; x2 int - coordonata pe axa OX al doilea vector
; y2 int - coordonata pe axa OY al doilea vector
(defun produsScalar(vector1_x vector1_y vector2_x vector2_y)
    (+ (* vector1_x vector2_x) (* vector1_y vector2_y))
)

; Problema 2
; MAX(a, b) = a; a > b
;             b; altfel
; a int - numar de comparat
; b int - numar de comparat
;                 -inf; n = 0
; maxim(l1..ln) = MAX(l1, maxim(l2..ln)); l1 - atom numeric
;                 maxim(l2..ln); l1 - atom
;                 MAX(maxim(l1), maxim(l2..ln)); altfel
; l1..ln list - lista din care se determina maximul
(defun maxim(l)
    (cond
        ((null l) -9999999)
        ((numberp (car l)) (MAX (car l) (maxim (cdr l))))
        ((atom (car l)) (maxim (cdr l)))
        (t (MAX (maxim (car l)) (maxim (cdr l))))
    )
)

; Problema 3
; ins(e, n, l1..lk) = (e l1..lk); n = 1
;                     l1 (+) ins(e, n - 1, l2..lk) altfel
; e int - numarul care sa se adauge in lista
; n int - pozitia pe care sa se adauge in lista
; l1..lk lista - lista in care sa se adauge numarul e
(defun ins(e n l)
    (cond
        ((= n 1) (cons e l))
        (t (cons (car l) (ins e (- n 1) (cdr l))))
    )
)

; insert(e, n, l1..lk) = vida; n = 0
;                        ins(e,n,l1..lk) (+) insert(e,n-1,l1..lk) altfel
; e int - numarul de adaugat in liste pe toate pozitiile
; n int - pozitiile pe care sa se adauge in lista
; l1..lk lista - lista in care se adauga pe fiecare pozitie numarul e
(defun insert(e n l)
    (cond
        ((= n 0) nil)
        (t (cons (ins e n l) (insert e (- n 1) l)))
    )
)

; inserare(e, l1..ln) = insert(e,n+1,l1..ln)
; e int - numarul de adaugat in lista pe fiecare pozitie
; l1..ln lista - lista in care se adauga elementul e pe fiecare pozitie
(defun inserare(e l)
    (insert e (+ (length l) 1) l)
)

; auxPem(e, l1..ln) = vida; n = 0
;                     inserare(e, l1) (+) auxPem(e, l2..ln); altfel
; e int - adauga elementul e in fiecare sublista din l1..ln
; l1..ln lista - lista cu liste
(defun auxPem(e l)
    (cond
        ((null l) nil)
        (t (append (inserare e (car l)) (auxPem e (cdr l))))
    )
)

;                     vida; n = 0
; permutari(l1..ln) = (l1); n = 1
;                     inserare(l1, l2); n = 2
;                     auxPem(l1, permutari(l2..ln)); altfel
; l1..ln - lista caruia sa ii se faca toate permutarile posibile
(defun permutari(l)
    (cond
        ((null l) nil)
        ((null (cdr l)) l)
        ((null (cddr l)) (inserare (car l) (cdr l)))
        (t (auxPem (car l) (permutari (cdr l))))
    )
)

; Problema 4
;                       t; n = 0
; lungimePara(l1..ln) = false; n = 1
;                       lungimePara(l3..ln); n > 2
; l1..ln lista - lista de verificat daca are numar par sau impar de elemente
(defun lungimePara(l)
    (cond
        ((null l) t)
        ((null (cdr l)) nil)
        (t (lungimePara(cddr l)))
    )
)