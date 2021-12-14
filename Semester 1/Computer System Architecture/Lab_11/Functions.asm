%ifndef __FUNC_ASM__
%define __FUNC_ASM__

; void get_digits(int length, char* string, char* digits_string)
; param: length - lungimea sirului string
; param: string - sirul din care extragem cifrele zecimale
; param: digits_string - sirul in care vom pune cifrele zecimale
get_digits:

    ; Luam parametrii de pe stiva
    mov ECX, [ESP + 4 * 1]
    mov ESI, [ESP + 4 * 2]
    mov EDI, [ESP + 4 * 3]
    
    xor EBX, EBX        ; In EBX vom retine cate cifre zecimale apar
    
    ; Parcurgem sirul string caracter cu caracter
    cld
    string_for:
    
        lodsb
        
        ; 0x30 - '0', 0x39 - '9'. Verificam caracterul de pe pozitia curenta sa fie cifra
        cmp AL, 0x30
        jb not_digit
        cmp AL, 0x39
        ja not_digit
            
            ; Daca e cifra il punem in sirul digits_string si incrementam numarul de cifre zecimale aparute
            inc EBX
            stosb
            
        not_digit:
        
    loop string_for
    
    ret 4 * 3

; void concatenate(char* conc, int length_1, char* digits_string_1, int length_2, char* digits_string_2)
; conc - sirul rezultat in care se vor concatena cele 2 siruri de cifre
; length_1 - lungimea sirului digits_string_1
; digits_string_1 - primul sir cu cifre
; length_2 - lungimea sirului digits_string_2
; digits_string_2 - al doilea sir cu cifre
concatenate:
    
    ; Luam sirul rezultat si primul sir de cifre de pe stiva
    mov EDI, [ESP + 4 * 1]
    mov ECX, [ESP + 4 * 2]
    mov ESI, [ESP + 4 * 3]
    
    ; Punem primul sir in sirul rezultat conc
    rep movsb
    
    ; Luam al doilea sir de cifre de pe stiva
    mov ECX, [ESP + 4 * 4]
    mov ESI, [ESP + 4 * 5]
    
    ; Punem al doilea sir in sirul rezultat conc dupa primul sir
    rep movsb
    
    ret 4 * 5
    
%endif