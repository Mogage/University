;                                 false; n = 0 vida
; nivelNodAux(l1..ln Nod Nivel) = Nivel;l1 = Nod, N = 1
;                                 false; n = 1, l1 != nod
;                                 nivelNodAux(l2 Nod Nivel+1); N = 2
;                                 nivelNodAux(l2 Nod Nivel+1) || nivelNodAux(l3 Nod Nivel+1); N = 3
; l1..ln lista - lista tip arbore in care se cauta
; Nod int - nodul caruia sa se afle nivelul in lista
; Nivel - nivelul curent pe care se afla un nod
(defun nivelNodAux(Arbore Nod Nivel)
    (cond
        ((null Arbore) nil)
        ((equal Nod (car Arbore)) Nivel)
        (t (OR (nivelNodAux (cadr Arbore) Nod (+ Nivel 1)) (nivelNodAux (caddr Arbore) Nod (+ Nivel 1))))
    )
)

; nivelNod(l1..ln Nod) = nivelNodAux(l1..ln Nod 0)
; l1..ln lista - lista tip arbore in care se cauta
; Nod int - nodul caruia sa se afle nivelul in lista
(defun nivelNod (Arbore Nod)
    (nivelNodAux Arbore Nod 0)
)