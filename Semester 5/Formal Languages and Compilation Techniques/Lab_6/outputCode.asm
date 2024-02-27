bits 32
global start

extern scanf
import scanf msvcrt.dll

extern printf
import printf msvcrt.dll

extern exit
import exit msvcrt.dll

segment data use32 class=data
	a dd 4
	b dd 0
	c dd 0
	format db "%d", 0

segment code use32 class=code
	start:
		push 4
		push 0
		pop dword [b]
		pop dword [a]
		push 4
		push 5
		push 6
		pop EDX
		pop EAX
		mul DL
		push EAX

		pop EDX
		pop EAX
		add AX, DX
		push EAX

		pop dword [a]

		push dword [a]
		push dword format
		call [printf]
		add ESP, 4 * 2

		push 4
		pop dword [a]

		push 2
		push 2
		pop EDX
		pop EAX
		mul DL
		push EAX

		push 4
		push 2
		pop EBX
		pop EAX
		mov DX, 0
		div WORD BX
		push EAX

		push 1
		pop EDX
		pop EAX
		sub AX, DX
		push EAX

		pop EDX
		pop EAX
		add AX, DX
		push EAX

		pop dword [b]

		push dword [a]
		push dword [b]
		pop EDX
		pop EAX
		add AX, DX
		push EAX

		pop dword [c]

		push dword a
		push dword format
		call [scanf]
		add ESP, 4 * 2

		push dword b
		push dword format
		call [scanf]
		add ESP, 4 * 2

		push dword [a]
		push dword format
		call [printf]
		add ESP, 4 * 2

		push dword [b]
		push dword format
		call [printf]
		add ESP, 4 * 2

		push dword [c]
		push dword format
		call [printf]
		add ESP, 4 * 2

		push dword 0
		call [exit]