%ifndef __FUNCS_ASM__
%define __FUNCS_ASM__

bits 32

global _get_chars

segment data public data use32
    string_address dd 0
    lower_bound_address dd 0
    upper_bound_address dd 0
    new_string_address dd 0
    
segment code public code use32

; void get_chars(char* string, char lower_bound, char upper_bound, char* new_string)
; string - sirul in care cautam caractere din intervalul [lower_bound, upper_bound]
; lower_bound - capatul inferior al intervalului in care se afla caracterele cautate
; upper_bound - capatul superior al intervalului in care se afla caracterele cautate
; new_string - noul sir in care punem caracterele care se afla in intervalul [lower_bound, upper_bound]
_get_chars:
    ; creare cadru de stiva pentru programul apelat
    push EBP
    mov EBP, ESP
    
    ; Luam parametrii de pe stiva
    mov EAX, [EBP + 4 * 2]
    mov [string_address], EAX
    mov EAX, [EBP + 4 * 3]
    mov [lower_bound_address], EAX
    mov EAX, [EBP + 4 * 4]
    mov [upper_bound_address], EAX
    mov EAX, [EBP + 4 * 5]
    mov [new_string_address], EAX
    
    ; Parcurgem sirul string caracter cu caracter
    mov ESI, [string_address]
    mov EDI, [new_string_address]
    cld
    .string_for:
        lodsb
        
        ; Verificam caracterul de pe pozitia curenta sa se afla in intervalul [lower_bound, upper_bound]
        cmp AL, [lower_bound_address]
        jb .not_in_between
        cmp AL, [upper_bound_address]
        ja .not_in_between
            
            ; Daca se afla in punem in sirul new_string
            stosb
        
        .not_in_between:
        ; Daca s-a ajung la caracterul 0 inseamna ca s-a terminat sirul de parcurs
        cmp AL, 0
        je .function_end
    
    jmp .string_for
    
    .function_end:
    stosb               ; Punem caracterul 0 la finalul noului sir format
    
    ; refacem cadrul de stiva pentru programul apelant
    mov ESP, EBP
    pop EBP
    
    ; mov EAX, new_string
    ret

%endif