bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit               ; tell nasm that exit exists even if we won't be defining it
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
                          ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions

; our data is declared here (the variables needed by our program)
segment data use32 class=data

    a db 01101100b
    b dw 1001101110111110b
    c dd 0
    
; our code starts here
segment code use32 class=code
    start:
        xor EAX, EAX
        xor EBX, EBX
        xor ECX, ECX
        xor EDX, EDX
        
        ; Cerinta:
        ;   Se da octetul A si cuvantul B. Sa se formeze dublucuvantul C:
        ;   - bitii 24-31 ai lui C sunt bitii lui A
        ;   - bitii 16-23 ai lui C sunt inversul bitilor din octetul cel mai putin semnificativ al lui B
        ;   - bitii 10-15 ai lui C sunt 1
        ;   - bitii 2-9 ai lui C sunt bitii din octetul cel mai semnificativ al lui B
        ;   - bitii 0-1 se completeaza cu valoarea bitului de semn al lui A
        
        ; Rezultatul o sa fie in DX:BX
        
        mov DH, [a]         ; Bitii 24-31 din DX:BX sunt a lui A
        
        mov AL, byte [b]    ; Izolam bitii din octelul cel mai putin semnificativ a lui B
        not AL              ; Inversam valoarea octetului
        mov DL, AL          ; Punem bitii in rezultat, reprezentand bitii 16-23 din DX:BX
        
        or  BH, 11111100b   ; Facem bitii 10-15 din rezultat sa aiba valoarea 1
        
        mov AL, byte [b+1]  ; Izolam bitii din octelul cel mai semnificativ a lui B
        mov CL, 2           
        rol AX, CL          ; Rotim 2 pozitii spre stanga
        or  BX, AX          ; Punem bitii in rezultat, reprezentand bitii 2-9 din DX:BX
        
        mov AX, 0           ; Golim AX
        mov AL, [a]         ; Izolam bitul de semn a lui A
        and AL, 10000000b
        mov CL, 6
        ror AL, CL          ; Rotim 6 pozitii spre dreapta
        or  BX, AX          ; Facem bitul 1 valoarea bitului de semn a lui A
        ror AL, 1           ; Rotim o pozitie spre dreapta
        or  BX, AX          ; Facem bitul 0 valoarea bitului de semn a lui A
        
        ; Punem rezultatul in C
        
        push DX
        push BX
        pop dword [c]
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
