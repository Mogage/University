bits 32

global start

extern exit, scanf, fopen, fclose, printf, fprintf

import scanf msvcrt.dll
import printf msvcrt.dll
import fprintf msvcrt.dll
import fopen msvcrt.dll
import fclose msvcrt.dll
import exit msvcrt.dll


segment data use32 class=data
    aux_var db 0                ; Variabila auxiliara ca sa ne asiguram ca se opreste parcurgerea unde trebuie
    file_name times 101 db 0
    file_name_length dd 0
    format_string db "%s", 0
    access_mode db "w", 0
    input_file dd 0
    lower_case times 101 db 0
    upper_case times 101 db 0
    
    A_MIC equ "a"
    A_MARE equ "A"
    Z_MIC equ "z"
    Z_MARE equ "Z"

segment code use32 class=code
    start:
        ; Se citeste de la tastatura (un cuvant) numele unui fisier
        ; Sa se scrie in fisier numele fisierului intai cu minuscule si dupa cu majuscule in ordine inversa
        
        
        ; Citim numele fisierului de la tastatura, CU TOT CU EXTENSIA .TXT
        push    file_name
        push    format_string
        call    [scanf]
        add     ESP, 4 * 2
        
        
        ; Daca nu este creat fisierul il cream noi si il deschidem
        push    access_mode
        push    file_name
        call    [fopen]
        add     ESP, 4 * 2
        
        ; Daca EAX e 0 a fost o eroare la deschiderea fisierului
        test    EAX, EAX
        jz      eroare
        
        mov     [input_file], EAX
        
        ; Aflam lungimea sirului pentru a il putea parcurge invers 
        mov     ESI, file_name
        cld
        find_length:
            lodsb
            
            cmp     AL, 0
            je      length_found
            
            inc     dword [file_name_length]
        jmp find_length
        
        length_found:
        dec     dword [file_name_length]        ; Decrementam lungimea astfel incat sa ne aflam pe ultimul caracter nenul
                                                ; din numele fisierului
        
        ; Eliminam extensia care s-a citit intrucat lucram doar cu numele fisierului
        mov     EBX, [file_name_length]
        mov     byte [EBX + file_name - 3], 0
        
        ; Punem adresa de la finalul sirului in ESI si il parcurgem in ordine inversa
        lea     ESI, [EBX + file_name - 4]
        mov     EDI, lower_case
        
        ; Parcurgem numele fisierului pentru a obtine literele mici
        file_name_for:
            ; Ne asiguram ca DF sa fie setat astfel incat ESI sa se decrementeze
            std
            lodsb
            
            cmp     AL, A_MIC
            jb      not_in_between
            cmp     AL, Z_MIC
            ja      not_in_between
                ; Ne asiguram ca DF sa nu fie setat ca sa putem adauga in sirul rezultat
                cld
                stosb
            
            not_in_between:
            
            ; Daca s-a ajung la ultimul caracter 0 atunci inseamna ca am terminat de parcurs numele fisierului
            cmp     AL, 0
            je      end_for
            
        jmp file_name_for
        
        ; Punem 0 la finalul sirului
        end_for:
        stosb
        
        ; Acelasi proces se intampla si pentru literele mari, se poate face o functie
        mov     EBX, [file_name_length]
        lea     ESI, [EBX + file_name - 4]
        mov     EDI, upper_case
        
        file_name_for_upper:
            std
            lodsb
            
            cmp     AL, A_MARE
            jb      not_in_between_upper
            cmp     AL, Z_MARE
            ja      not_in_between_upper
                cld
                stosb
            
            not_in_between_upper:
            
            cmp     AL, 0
            je      end_for_upper
            
        jmp file_name_for_upper
        
        end_for_upper:
        stosb
        
        ; Scriem in fisier cele 2 siruri
        push    lower_case
        push    format_string
        push    dword [input_file]
        call    [fprintf]
        add     ESP, 4 * 3
        
        push    upper_case
        push    format_string
        push    dword [input_file]
        call    [fprintf]
        add     ESP, 4 * 3
        
        ; Inchidem fisierul
        push    dword [input_file]
        call    [fclose]
        add     ESP, 4
        
        eroare:
        
        push    dword 0
        call    [exit]