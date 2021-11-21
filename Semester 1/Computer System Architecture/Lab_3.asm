bits 32 ; assembling for the 32 bits architecture

; declare the EntryPoint (a label defining the very first instruction of the program)
global start        

; declare external functions needed by our program
extern exit               ; tell nasm that exit exists even if we won't be defining it
import exit msvcrt.dll    ; exit is a function that ends the calling process. It is defined in msvcrt.dll
                          ; msvcrt.dll contains exit, printf and all the other important C-runtime specific functions

; our data is declared here (the variables needed by our program)
segment data use32 class=data
    ; Adunări, scăderi: 10 (prima parte) 11 (a doua parte)    
    
    ; 10
    ; a db 2h
    ; b dw 8888h
    ; c dd 1110h
    ; d dq 0FFFFFFFFh
    
    ; 11
    ; a db 1h
    ; b dw 0FFFFh
    ; c dd 20h
    ; d dq 10000h

    ; Înmulțiri, împărțiri: 20 
    
    a dw 3h
    b db 7h
    e dd 1000h
    x dq 10000h
    
    
; our code starts here
segment code use32 class=code
    start:
        xor EAX, EAX
        xor EBX, EBX
        xor ECX, ECX
        xor EDX, EDX
        
        ; Adunări, scăderi: 10 (prima parte) 11 (a doua parte)    
        
        ; 10 - (a+d+d)-c+(b+b) = (2 + FFFF FFFF + FFFF FFFF) - 1110 + (8888 + 8888) = 2 0001 0000
        ; Spatii lasate la fiecare 4 bytes pentru o citire mai usoara
        ; CF = Carry Flag
        ; mov AL, [a] ; AL = a
        
        ; add EAX, dword [d]      ; EAX = EAX + cei 4 bytes low a lui d
        ; adc EDX, dword [d + 4]  ; EDX = EDX + cei 4 bytes high a lui d + CF
        
        ; add EAX, dword [d]      ; EBX = cei 4 bytes low a lui d
        ; adc EDX, dword [d + 4]  ; ECX = cei 4 bytes high a lui d
        
        ; ; EDX:EAX = a + d + d
        
        ; sub EAX, [c]            ; EAX = EAX - c = (a + d + d) - c
        ; sbb EDX, 0              ; EDX = EDX - CF, pentru cazul in care se imprumuta pentru scaderea lui c

        ; MOV BX, [b]             ; BX = b
        ; add BX, [b]             ; BX = BX + b = b + b
        ; adc CX, 0               ; CX = CX + CF, in cazul in care depasteste b + b
        
        ; push CX
        ; push BX
        ; pop EBX
        
        ; ; EBX = CX:BX = b + b
        
        ; add EAX, EBX            ; EAX = EAX + EBX = (a + d + d) - c + (b + b)
        ; adc EDX, 0              ; EDX = EDX + CF
        
        ; Rezultat este in EDX:EAX
        
        ; ------------------------------------------------------------------------------
        
        ; 11 - d-(a+b+c)-(a+a) = 1 0000 - (1 + FFFF + 20) - (1 + 1) = FFDE
        
        ; mov AL, [a]     ; AL = a
        ; cbw             ; conversie cu semn de la AL la AX
        ; cwd             ; conversie cu semn de la AX la DX:AX
        
        ; push DX 
        ; push AX
        ; pop EBX
        
        ; ; EBX = DX:AX = a
        
        ; mov AX, [b]     ; AX = b
       
        ; cwde ; conversie cu semn AX -> EAX
        
        ; add EAX, EBX    ; EAX = EAX + EBX = b + a
        
        ; cdq ; conversie cu semn de la EAX la EDX:EAX
        
        ; clc
        ; add EAX, [c]    ; EAX = EAX + c = a + b + c
        ; adc EDX, 0
        
        ; mov EBX, dword [d]          ; EBX = cei 4 bytes low din d
        ; mov ECX, dword [d + 4]      ; ECX = cei 4 bytes high din d
        
        ; ; ECX:EBX = d
        
        ; sub EBX, EAX        ; EBX = EBX - EAX 
        ; sub ECX, EDX        ; ECX = ECX - EDX
        
        ; ; ECX:EBX = d - (a + b + c)
        
        ; xor EAX, EAX        ; EAX = 0
        ; xor EDX, EDX        ; EDX = 0
        
        ; mov AL, [a]         ; AL = a
        ; cbw     ; conversie cu semn de la AL la AX
        ; add AL, [a]         ; AL = AL + a = a + a
        
        ; cwde    ; conversie cu semn de la AX la EAX
        ; cdq     ; conversie cu semn de la EAX la EDX:EAX
        
        ; sub EBX, EAX        ; EBX = EBX - EAX
        ; sbb ECX, EDX        ; ECX = ECX - EDX - CF
        
        ; ; ECX:EBX = d - (a + b + c) - (a + a)
        
        ; mov EAX, EBX        ; EAX = EBX
        ; mov EDX, ECX        ; EDX = ECX
        
        ; EDX:EAX = ECX:EBX, rezultatul fiind in EDX:EAX
        
        ; ------------------------------------------------------------------------------
        
        ; Înmulțiri, împărțiri: 20  
        
        ; 20 - x-b+8+(2*a-b)/(b*b)+e = 1 0000 - 7 + 8 + (2 * 3 - 7) / (7 * 7) + 1000 = 53A 882A
        ; Interpretare fara semn
        
        ; mov AL, [b]     ; AL = b
        ; mul AX          ; DX:AX = AX * AX = b * b
        ; mov CX, AX      ; CX = AX = b * b
        
        ; mov AX, 2       ; AX = 2
        ; mul word [a]    ; DX:AX = AX * a = 2 * a
        
        ; ; EAX = DX:AX = 2 * a
        ; push DX
        ; push AX
        ; pop EAX

        ; mov BL, [b]     ; BL = b
        ; sub EAX, EBX    ; EAX = EAX - EBX = 2 * a - b
        
        ; div ECX         ; EAX = EDX:EAX / ECX = (2 * a - b) / (b * b)
        ; xor EDX, EDX    ; EDX = 0, incat nu avem nevoie de rest
        
        ; add EAX, [x]        ; EAX = EAX + cei 4 bytes low din x = x + (2 * a - b) / (b * b)
        ; adc EDX, [x + 4]    ; EDX = EDX + cei 4 bytes high din x + CF
        
        ; add EAX, [e]    ; EAX = EAX + e = x + (2 * a - b) / (b * b) + e
        ; adc EDX, 0      ; EAX = EAX + 0 + CF
        
        ; ; EDX:EAX = x - b + (2 * a - b) / (b * b) + e
        ; sub EAX, EBX    ; EAX = EAX - EBX
        ; sbb EDX, 0      ; EDX = EDX - 0 - CF
        
        ; add EAX, 8      ; EAX = EAX + 8
        ; adc EDX, 0      ; EDX = EDX - 0 - CF 
        
        ; EDX:EAX = x - b + 8 + (2 * a - b) / (b * b) + e

        ; ------------------------------------------------------------------------------
        
        ; 20 - x-b+8+(2*a-b)/(b*b)+e = 1 0000 - 7 + 8 + (2 * 3 - 7) / (7 * 7) + 1000 = 1 1001
        ; Interpretare cu semn
        
        mov AL, [b]     ; AL = b
        imul AL         ; AX = AL * AL = b * b
        mov CX, AX      ; CX = AX = b * b
        
        mov AX, 2       ; AX = 2
        imul word [a]   ; DX:AX = 2 * a
        
        ; EAX = DX:AX = 2 * a
        push DX
        push AX
        pop EAX

        mov BL, [b]     ; BL = b
        sub EAX, EBX    ; EAX = EAX - EBX = 2 * a - b
        cdq             ; conversie cu semn de la EAX la EDX:EAX
        
        idiv ECX        ; EAX = EDX:EAX / ECX = EAX = (2 * a - b) / (b * b), EDX = EDX:EAX % ECX
        xor EDX, EDX    ; EDX = 0 fiindca ne folosim doar de catul de la impartire
        
        ; EDX:EAX = x + (2 * a - b) / (b * b)
        add EAX, [x]        ; EAX = EAX + cei 4 biti low din x 
        adc EDX, [x + 4]    ; EDX = EDX + cei 4 biti high din x
        
        ; EDX:EAX = x + (2 * a - b) / (b * b) + e
        add EAX, [e]    ; EAX = EAX + e = x + (2 * a - b) / (b * b) + e
        adc EDX, 0    ; EAX = EAX + 0 + CF, daca de exemplu x = FFFF FFFF iar e = 1 rezultatul din EDX:EAX trebuie sa fie 1 0000 0000
        
        ; EDX:EAX = x - b + (2 * a - b) / (b * b) + e
        sub EAX, EBX    ; EAX = EAX - EBX
        cdq             ; conversie cu semn de la EAX la EDX:EAX
        
        add EAX, 8
        cdq             ; conversie cu semn de la EAX la EDX:EAX
        ; EDX:EAX = x - b + 8 + (2 * a - b) / (b * b) + e
    
        ; exit(0)
        push    dword 0      ; push the parameter for exit onto the stack
        call    [exit]       ; call exit to terminate the program
