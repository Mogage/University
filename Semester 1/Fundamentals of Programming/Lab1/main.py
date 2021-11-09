# Cel mai mare divizor comun

def cmmdc(a, b):
    if (a < 0 or b < 0):
        return "Valorile trebuie sa fie pozitive"
    while(b):
        c = a % b
        a = b
        b = c
    return a

x = int(input("Introduceti primul numar: "))
y = int(input("Introduceti al doilea numar: "))
print(cmmdc(x,y))

# Suma a n numere
'''
n = int(input("Introduceti n:"))
s = 0
for i in range():
    x = input("Numarul " + str(i + 1) + ":")
    s += int(x)
print("Suma celor " + n + " numere este " + str(s))
'''

# Verifica daca un numar este prim
import math
'''
def prim(n):
    if (n == 2 or n == 3): return "Numarul este prim"
    if (n % 2 == 0 or n % 3 == 0 or n < 2): return "Numarul nu este prim"
    i = 5
    while(i * i <= n):
        if(n % i == 0 or n % (i+2) == 0):
            return "Numarul nu este prim"
        i += 6
    return "Numarul este prim"

n = int(input("Introduceti numarul: "))
print(prim(n))
'''
