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
    
    S: db '+', '4', '2', 'a', '8', '4', 'X', '5'
    
    l equ $ - S
    
    d times l db 0
    
; our code starts here
segment code use32 class=code
    start:
        xor     EAX, EAX    ; EAX = 0
        xor     EBX, EBX    ; EBX = 0
        xor     EDX, EDX    ; EDX = 0
        
        ; Cerinta:
        ;    Se da un sir de caractere S. Sa se construiasca sirul D care sa contina toate caracterele cifre din sirul S.
        ; Exemplu:
        ;   S: '+', '4', '2', 'a', '8', '4', 'X', '5'
        ;   D: '4', '2', '8', '4', '5'
        
        mov     ECX, l      ; ECX = l
        xor     ESI, ESI    ; ESI = 0
        xor     EDI, EDI    ; EDI = 0
        
        mov     BL, '0'  ; BL = '0', 30 in ASCII
        mov     DL, '9'  ; DL = '9', 39 in ASCII
        
        jecxz sfarsit       ; Daca lungimea sirului S e 0, adica daca ECX e 0 sare la finalul programului
        
    repeta:         ; Start instructiune repetitiva
        
        mov     AL, byte [S + ESI]  ; AL = un caracter din sirul S de pe poz ESI
        inc     ESI                 ; ESI = ESI + 1
        cmp     AL, BL              ; Compara caracterul curent cu '0'
        jl continua                 ; Daca este mai mic decat '0' nu este numar si nu se adauga in D si se sare la finalul instructiunii repetitive
        cmp     AL, DL              ; Compara caracterul curent cu '9'
        jg continua                 ; Daca este mai mare decat '9' nu este numar si nu se adauga in D si se sare la finalul instructiunii repetitive
        
        mov     byte [d + EDI], AL  ; Se adauga numarul in D
        inc     EDI                 ; EDI = EDI + 1
        
    continua:       ; Se continua pentru cazul in care caracterul nu e numar
    
    loop repeta     ; Continuare intructiune repetitiva
        
    sfarsit:        ; Terminare program
        
        ; Afisare d
        push    dword d
        call    [printf]
        add     esp, 4
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
