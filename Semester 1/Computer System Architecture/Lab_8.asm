bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit               ; tell nasm that exit exists even if we won't be defining it
extern scanf, printf
extern fopen, fclose, fread, fprintf
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
                          ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions
import scanf msvcrt.dll                  
import printf msvcrt.dll                  
import fopen msvcrt.dll                  
import fclose msvcrt.dll                  
import fread msvcrt.dll                  
import fprintf msvcrt.dll                  

; our data is declared here (the variables needed by our program)
segment data use32 class=data
    ; ------------------------------Problema cu functii-----------------------------------------
    ; number dd 0x80000000        ; Variabila cu care facem citirea numerelor de la tastatura
    ; maxim dd 0x80000000         ; Variabila in care vom pune cel mai mare numar
    ; int_format db "%d", 0       ; Format pentru int
    ; read_message db "Introduceti un numar: ", 0         ; Mesaj pentru citirea numerelor
    ; output_message db "Cel mai mare numar este: %d", 0   ; Mesaj pentru afisarea celui mai mare numar
    
    ; ------------------------------Problema cu fisiere-----------------------------------------
    file_name db "input.txt", 0
    read_acces db "r", 0
    input resd 1
    
    len equ 100
    buffer times (len + 1) db 0
    
    vowels db 'a', 'e', 'i', 'o', 'u', 0
    number_of_vowels dd 0
    output_message db "Numarul de vocale din fisier este: %d", 0
; our code starts here
segment code use32 class=code
    start:
        ; ------------------------------Problema cu functii-----------------------------------------
        
        ; Cerinta:
        ;   Se citesc de la tastatura numere (in baza 10) pana cand se introduce cifra 0.
        ;   Determinaţi şi afişaţi cel mai mare număr dintre cele citite.
        
        ; Determinam maximul dintre numere in timpul citirii si nu luam in considerare cifra 0
        ; Consideram ca se citeste cel putin un numar diferit de 0
        ; input:
            ; ; Vedem daca numarul citit curent este mai mare decat cele citite anterior
            ; mov EAX, dword [number]
            ; cmp EAX, dword [maxim]
            
            ; ; Daca este mai mic sau egal inseamna ca nu trebuie sa actualizam maximul
            ; jle lower
                
                ; mov dword [maxim], EAX      ; Actualizam rezultatul
            
            ; lower:
            
            ; ; Afisare mesaj pentru citirea unui numar
            ; push dword read_message
            ; call [printf]
            ; add ESP, 4
            
            ; ; Citirea numarului de la tastatura
            ; push dword number
            ; push dword int_format
            ; call [scanf]
            ; add ESP, 4 * 2
            
            ; ; Daca numarul nu e 0 continuam citirea
            ; cmp dword [number], 0
        ; jnz input

        ; ; Afisam numarul maxim dintre cele citite
        ; push dword [maxim]
        ; push dword output_message
        ; call [printf]
        ; add ESP, 4 * 2
        
        ; ------------------------------Problema cu fisiere-----------------------------------------
        
        ; Cerinta:
        ;   Se da un fisier text. Sa se citeasca continutul fisierului, sa se contorizeze numarul de vocale si sa se afiseze aceasta valoare.
        ;   Numele fisierului text este definit in segmentul de date.
        
        ; Apelam functia fopen pentru a deschide fisierul de unde citim datele
        push dword read_acces
        push dword file_name
        call [fopen]
        add ESP, 4 * 2
        
        ; Daca s-a produs o eroare sarim la final
        cmp EAX, 0
        je final
        
        mov dword [input], EAX  ; Salvam descriptorul fisierului
        
        read_all_file:
            ; Apelam functia fread pentru a citi o parte (100 caractere) din fisier
            push dword [input]
            push dword len
            push dword 1
            push dword buffer
            call [fread]
            add ESP, 4 * 4
            
            ; Daca numarul de caractere citit este 0, am terminat de parcurs fisierul
            cmp EAX, 0          ; In EAX se afla cate caractere au fost citite
            je end_of_file
            
            mov ESI, buffer ; ESI = adresa de inceput a bucatii citite din fisier
            mov ECX, EAX    ; Mutam in ECX numarul de caractere cititie pentru a parcurce buffer-ul caracter cu caracter intr-un loop
            
            ; Parcurgem buffer-ul
            parcurgere_text:
                
                lodsb       ; Aducem in AL un caracter din buffer
              
                ; Verificam daca respectivul caracter e vocala
                mov EDI, 0
                check_vowel:
                    cmp AL, byte [vowels + EDI]
                    jne not_vowel
                        inc dword [number_of_vowels]    ; Daca e vocala incrementam rezultatul
                    not_vowel:
                    
                    inc EDI
                    cmp EDI, 5
                jl check_vowel
            
            loop parcurgere_text
        
        jmp read_all_file
        
        end_of_file:
            ; Afisam numarul de vocale
            push dword [number_of_vowels]
            push dword output_message
            call [printf]
            add ESP, 4 * 2
            
            ; Apelam functia fclose inchide fisierul de unde am citit datele
            push dword [input]
            call [fclose]
            add ESP, 4
        
        final:
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
