;               false; daca n = 0
; sau(l1..ln) = adevarat; daca l1 este adevarat
;               sau(l2..ln); altfel
; l1..ln lista - lista cu valori adevarat sau fals care sa se faca
(defun sau(lista)
    (cond
        ((null lista) nil)
        ((car lista) t)
        (t (sau (cdr lista)))
    )
)

;                          false; daca n = 0
;apartenenta(l1..ln Nod) = adevarat; daca l1=Nod
;                          sau(apartenenta(l2), ..., apartenenta(ln))
; l1..ln lista - lista de tip arbore in care se cauta nodul Nod
; Nod int - atomul de cautat in lista daca exista
(defun apartenenta(Arbore Nod)
    (cond
        ((null Arbore) nil)
        ((equal Nod (car Arbore)) t)
        (t (funcall #'sau 
                (mapcar #'(lambda (Arbore) (apartenenta Arbore Nod)) (cdr Arbore))
           )
        )   
    )
)