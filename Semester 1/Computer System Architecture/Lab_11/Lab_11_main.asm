bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit, printf       ; tell nasm that exit exists even if we won't be defining it
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
import printf msvcrt.dll  ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions

%include "Functions.asm"
; our data is declared here (the variables needed by our program)
segment data use32 class=data
    string_1 db "fds';123fs';456;dsd'7", 0
    LENGTH_STRING_1 equ $ - string_1 - 1
    string_2 db "$@#FDS765d]'432sna10s", 0
    LENGTH_STRING_2 equ $ - string_2 - 1
    number_of_digits_1 dd 0
    number_of_digits_2 dd 0
    digits_string_1 times (LENGTH_STRING_1 + 1) db 0
    digits_string_2 times (LENGTH_STRING_2 + 1) db 0
    concatenation times (LENGTH_STRING_1 + LENGTH_STRING_2) db 0
    format_concatenation_1 db "Rezultatul concatenarii cifrelor din primul sir cu cele din al doilea sir este: %s", 13, 10, 0
    format_concatenation_2 db "Rezultatul concatenarii cifrelor din al doilea sir cu cele din primul sir este: %s", 13, 10, 0

; our code starts here
segment code use32 class=code
    start:
        
        ; Extragem cifrele din sirul string_1 si le punem in digits_string_1
        ; get_digits(LENGTH_STRING_1, string_1, digits_string_1)
        push dword digits_string_1
        push dword string_1
        push dword LENGTH_STRING_1
        call get_digits
        mov [number_of_digits_1], EBX
        
        ; Extragem cifrele din sirul string_2 si le punem in digits_string_2
        ; get_digits(LENGTH_STRING_2, string_2, digits_string_2)
        push dword digits_string_2
        push dword string_2
        push dword LENGTH_STRING_2
        call get_digits
        mov [number_of_digits_2], EBX
        
        ; Concatenam primul sir de cifre cu al doilea, in aceasta ordine
        ; concatenate(concatenation, number_of_digits_1, digits_string_1, number_of_digits_2, digits_string_2)
        push dword digits_string_2
        push dword [number_of_digits_2]
        push dword digits_string_1
        push dword [number_of_digits_1]
        push dword concatenation
        call concatenate
        
        ; Afisam sirul concatenat
        ; printf(format_concatenation_1, concatenation)
        push concatenation
        push format_concatenation_1
        call [printf]
        add ESP, 4 * 2
        
        ; Concatenam al doilea sir de cifre cu primul, in aceasta ordine
        ; concatenate(concatenation, number_of_digits_2, digits_string_2, number_of_digits_1, digits_string_1)
        push dword digits_string_1
        push dword [number_of_digits_1]
        push dword digits_string_2
        push dword [number_of_digits_2]
        push dword concatenation
        call concatenate
        
        ; Afisam sirul concatenat
        ; printf(format_concatenation_1, concatenation)
        push concatenation
        push format_concatenation_2
        call [printf]
        add ESP, 4 * 2
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
