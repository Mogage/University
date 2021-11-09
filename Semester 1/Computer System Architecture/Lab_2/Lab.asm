bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit               ; tell nasm that exit exists even if we won't be defining it
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
                          ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions

; our data is declared here (the variables needed by our program)
segment data use32 class=data

    ; Adunări, scăderi: 22 (prima parte) 21 (a doua parte) 
    
    ; 22
    ; a DB 10
    ; b DB 20
    ; c DB 5
    ; d DB 15
    
    ; 21
    ; a DW 5
    ; b DW 5
    ; c DW 1
    ; d DW 2
    
    
    ; Înmulțiri, împărțiri: 27 (prima parte) 21 (a doua parte)
    
    ; 27
    ; a DB 10
    ; b DB 6
    ; c DB 1
    ; d DW 14
    
    ; 21
    ; a DB 2
    ; b DB 2
    ; c DB 1
    ; d DB 3
    ; e DW 2
    ; f DW 8
    ; g DW 4
    ; h DW 1


; our code starts here
segment code use32 class=code
    start:
        mov EAX, 0 ; Pentru usurinta in debugging
        
        ; Exercitii simple: 15 - 6*3
        
        ; mov AL, 6 ; AL = 6
        ; mov AH, 3 ; AH = 3
        ; mul AH    ; AX = AL * AH = 6 * 3
        
        
        ; Adunări, scăderi: 22 (prima parte) 21 (a doua parte) 
        
        ; 22 - (a+b+b)-(c+d) : (10+20+20)-(5+15) = 30 (1D)
        ; mov AL, [a] ; AL = a
        ; add AL, [b] ; AL = AL + b = a + b
        ; add AL, [b] ; AL = AL + b = a + b + b
        ; mov BL, [c] ; BL = c
        ; add BL, [d] ; BL = BL + d = c + d
        ; sub AL, BL  ; AL = AL - BL = (a + b + b) - (c + d)
        
        ; 21 - a-c+d-7+b-(2+d) : 5-1+2-7+5-(2+2) = 0
        ; mov AX, [a] ; AX = a
        ; sub AX, [c] ; AX = AX - c = a - c
        ; add AX, [d] ; AX = AX + d = a - c + d
        ; sub AX, 7   ; AX = AX - 7 = a - c + d - 7
        ; add AX, [b] ; AX = AX + b = a - c + d - 7 + b
        ; mov BX, 2   ; BX = 2
        ; add BX, [d] ; BX = BX + d = 2 + d
        ; sub AX, BX  ; AX = AX - BX = a - c + d - 7 + b - (2 + d)
        
        
        ; Înmulțiri, împărțiri: 27 (prima parte) 21 (a doua parte)
        
        ; 27 - d/[(a+b)-(c+c)] : 14/[(10+6)-(1+1)] = 1
        ; mov AX, [d] ; AX = d
        ; mov BL, [a] ; BL = a
        ; add BL, [b] ; BL = BL + b = a + b
        ; mov BH, [c] ; BH = c
        ; add BH, [c] ; BH = BH + c = c + c
        ; sub BL, BH  ; BL = BL - BH = (a + b) - (c + c)
        ; div BYTE BL      ; AL = AX / BL = d / [(a + b) - (c + c)] , AH = AX % BL = d % [(a + b) - (c + c)]
        
        ; 21 - (f*g-a*b*e)/(h+c*d) : (8*4-2*2*2)/(1+1*3) = 6
        ; mov AX, [f] ; AX = f
        ; mov BX, [g] ; BX = g
        ; mul BX      ; DX:AX = AX * BX = f * g
        
        ; push  DX    ; se pune pe stiva partea high din double word-ul DX:AX
        ; push  AX    ; se pune pe stiva partea low din double word-ul DX:AX
        ; pop   EBX   ; EBX = DX:AX = f * g
        
        ; mov AL, [a] ; AL = a
        ; mov AH, [b] ; AH = b
        ; mul AH      ; AX = AL * AH = a * b
        ; mov CX, [e] ; CX = e
        ; mul CX      ; DX:AX = AX * CX = a * b * e
        
        ; push  DX    ; se pune pe stiva partea high din double word-ul DX:AX
        ; push  AX    ; se pune pe stiva partea low din double word-ul DX:AX
        ; pop   ECX   ; ECX = DX:AX = a * b * e
        
        ; sub EBX, ECX; EBX = EBX - ECX = f * g - a * b * e
        
        ; mov AL, [c] ; AL = c
        ; mov AH, [d] ; AH = d
        ; mul AH      ; AX = AL * AH = c * d
        ; add AX, [h] ; AX = AX + h = c * d + h
        ; mov CX, AX  ; CX = AX = c * d + h
        
        ; push EBX    ; se pune pe stiva double word-ul EBX
        ; pop  AX     ; se scoate partea low din EBX
        ; pop  DX     ; se scoate partea high din EBX
        
        ; div WORD CX ; AX = DX:AX / CX = (f * g - a * b * e) / (h + c * d) , DX = DX:AX % CX = (f * g - a * b * e) % (h + c * d)
        
        
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
