bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit               ; tell nasm that exit exists even if we won't be defining it
extern scanf, printf
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
                          ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions
import scanf msvcrt.dll                  
import printf msvcrt.dll                  

; our data is declared here (the variables needed by our program)
segment data use32 class=data
    number dd 0x80000000        ; Variabila cu care facem citirea numerelor de la tastatura
    maxim dd 0x80000000         ; Variabila in care vom pune cel mai mare numar
    int_format db "%d", 0       ; Format pentru int
    read_message db "Introduceti un numar: ", 0         ; Mesaj pentru citirea numerelor
    output_message db "Cel mai mare numar ete: %d", 0   ; Mesaj pentru afisarea celui mai mare numar
; our code starts here
segment code use32 class=code
    start:
        
        ; Cerinta:
        ;   Se citesc de la tastatura numere (in baza 10) pana cand se introduce cifra 0.
        ;   Determinaţi şi afişaţi cel mai mare număr dintre cele citite.
        
        ; Determinam maximul dintre numere in timpul citirii si nu luam in considerare cifra 0
        ; Consideram ca se citeste cel putin un numar diferit de 0
        input:
            ; Vedem daca numarul citit curent este mai mare decat cele citite anterior
            mov EAX, dword [number]
            cmp EAX, dword [maxim]
            
            ; Daca este mai mic sau egal inseamna ca nu trebuie sa actualizam maximul
            jle lower
                
                mov dword [maxim], EAX      ; Actualizam rezultatul
            
            lower:
            
            ; Afisare mesaj pentru citirea unui numar
            push dword read_message
            call [printf]
            add ESP, 4
            
            ; Citirea numarului de la tastatura
            push dword number
            push dword int_format
            call [scanf]
            add ESP, 4 * 2
            
            ; Daca numarul nu e 0 continuam citirea
            cmp dword [number], 0
        jnz input

        ; Afisam numarul maxim dintre cele citite
        push dword [maxim]
        push dword output_message
        call [printf]
        add ESP, 4 * 2
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
