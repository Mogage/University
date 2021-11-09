bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit, printf            ; tell nasm that exit exists even if we won't be defining it
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
import printf msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
                          ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions

; our data is declared here (the variables needed by our program)
segment data use32 class=data
    
    ; Cerinta:
    ;   Dandu-se un sir de cuvinte, sa se calculeze cel mai lung subsir de cuvinte ordonate crescator din acest sir.
    
    sir: dw 250h, 251h, 252h   ; Sirul de cuvinte din care se cauta 
    
    number_elements equ ($ - sir) / 2   ; Numarul de cuvinte din sir
    
    rez times number_elements dw 0      ; Sirul in care urmeaza sa fie pus rez
    
    first_pos db 0      ; Pozitia de pe care incepe subsirul
    second_pos db 0     ; Pozitia pe care se termina subsirul
    aux_length db 1     ; Var auxiliara pentru calcularea subsirului de lungime maxima
    max_length db 1     ; Lungimea maxima a subsirului
    
    format db "%x ", 0  ; Format pentru afisarea nr in baza 16
    
; our code starts here
segment code use32 class=code
    start:
        xor EAX, EAX
        xor EBX, EBX
        xor ECX, ECX
        xor EDX, EDX
        
        mov ESI, sir    ; ESI = adresa primului cuvant
        mov EDI, sir
        add EDI, 2      ; EDI = adresa celui al doilea cuvant
        mov EBX, 1      ; Prin EBX vom memora pozitia cuvantului actual din sir
        mov ECX, number_elements - 1    ; Vom parcurge sirul intr-o bucla loop cu number_elements - 1 iteratii
        
    repeta:
        cmpsw                       ; Comparam cuvintele de pe pozitii consecutive
        ja descrescator             ; Daca sir[BL - 1] > sir[BL] inseamna ca nu e crescator si sarim
        
        inc byte [aux_length]       ; Altfel crestem valoare lungimii auxiliare cu 1
        jmp crescator               ; si sarim peste operatia din cazul sir[i] > sir[i + 1]
    descrescator:                   ; Daca sir[BL - 1] > sir[BL] putem zice ca la pasul curent lungimea auxiliara a subsirului
        mov byte [aux_length], 1    ; este 1
    crescator:
        mov AL, [aux_length]        ; AL = aux_length
        mov AH, [max_length]        ; AH = max_length
        cmp AL, AH                  ; Verificam daca s-a gasit o lungime auxiliara mai mare decat cea maxima
        jbe continua                ; Daca aux_length <= max_length inseamna ca nu s-a gasit un subsir cu lungime 
                                    ; mai mare si sasim 
        mov byte [second_pos], BL   ; second_pos = BL
        
        ; first_pos = BL - aux_length + 1
        mov AL, BL                  ; AL = BL
        sub AL, byte [aux_length]   ; AL = AL - aux_length = BL - aux_length
        inc AL                      ; AL = AL + 1 = BL - aux_length + 1
        mov byte [first_pos], AL    ; first_pos = AL
        
        ; max_length = aux_length
        mov AL, byte [aux_length]
        mov byte [max_length], AL
        
    continua:
        inc BX                      ; BX = BX + 1 | Trecem la urmatorul element
        
        loop repeta                 ; Daca mai sunt cuvinte de parcus, se reia ciclul
        
        mov BL, byte [max_length]   ; Punem in BL lungimea subsirului
        
        ; Punem in ESI adresa cu care incepe primul element al subsirului
        xor EAX, EAX                ; EAX = 0
        mov AL, 2                   ; AL = 2
        mul byte [first_pos]        ; AX = 2 * first_pos | fiindca se folosesc cuvinte si se merge din 2 in 2
        mov ESI, sir                ; ESI = adresa primului cuvant al sirului
        add ESI, EAX                ; ESI = adresa primului cuvant al subsirului
        
    afisare:
        lodsw                       ; In AX vom avea cuvantul curent al subsirului
        
        mov word [rez + EDX], AX    ; Adaugam la rezultat cuvantul curent al subsirului
        add EDX, 2                  ; EDX = EDX + 2
        
        ; Afisare in consola
        push EAX
        push format
        call [printf]
        add ESP, 8
        
        dec BL                      ; BX = BX - 1
        jnz afisare                 ; Daca mai sunt cuvinte in subsir, se reia bucla
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
