# Problema 11
'''
            Cerinta

    11. Numerele n1 si n2 au proprietatea P daca scrierile lor in baza 10 conțin
        aceleași cifre (ex. 2113 si 323121). Determinați daca doua numere naturale
        au proprietatea P.

'''
def problema_11():
    # Citirea celor 2 numere pentru care verificam proprietatea P
    numarul1 = int(input ( "Introduceti primul numar: " ))
    numarul2 = int(input ( "Introduceti al doilea numar: " ))

    if (numarul1 < 0):
        numarul1 = -numarul1

    if (numarul2 < 0):
        numarul2 = -numarul2
    '''
        2 liste de aparitie in care lista[i] = True daca numarul contine cifra i, False atlfel
        verifica = True daca numerele au proprietatea P, False altfel
    '''
    lista1 = [0 for i in range(10)]  # echivalent cu lista1 = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lista2 = [0 for i in range(10)]  #                          0  1  2  3  4  5  6  7  8  9
    verifica = True

    # Parcurgerea numerelor si notarea aparitiei fiecarei cifre
    while numarul1 != 0:
        lista1[numarul1 % 10] = True
        numarul1 = numarul1 // 10

    while numarul2 != 0:
         lista2[numarul2 % 10] = True
         numarul2 = numarul2 // 10

    # Verificarea proprietatii P (daca cele 2 numere contin aceleasi cifre in baza 10)
    # prin parcugerea celor 2 liste simultan si compararea existentei acelorasi cifre
    for i, j in zip (lista1, lista2):
        if(i != j):
            verifica = False
            break

    # Afisarea rezultatului
    if(verifica): print("Numerele au proprietatea P.")
    else: print("Numerele nu au proprietatea P.")

'''                       ///////////////////////                   '''

# Problema 12

'''
                Cerinta

    12. Determinati al n-lea element al șirului 1,2,3,2,5,2,3,7,2,3,2,5,...
        obținut din șirul numerelor naturale prin înlocuirea numerelor compuse prin
        divizorii lor primi, fără a retine termenii șirului.

'''
def eratostene(numere_prime, e):
    '''
        Functie in care aflam numerele prime pana la 1000000 intrucat nu se cunoaste cat de mare poate fi n
    :param numere_prime: de tip lista in care introducem numerele prime
    :param e: e[x] = True daca x este prim si False atfel
    '''
    
    e[0] = e[1] = False
    index1 = 2
    while(index1 * index1 <= 1000000):
        if e[index1] == True:
            for index2 in range(index1 ** 2, 1000000, index1):
                e[index2] = False
        index1 += 1
    numere_prime.append(2)
    for index in range(3, 1000000, 2):
        if e[index] == True:
            numere_prime.append(index)

def problema_12(indice):
    '''
            Gasirea al n-lea element al sirului prin inlocuirea numerelor compuse prin divizorii lor primi
            /// indice - Indicele elementului citit de la tastatura (luat ca parametru pentru a putea face functia de teste) ///
            elemente - index pentru a parcurge numerele
            rezultat - variabila pe care o actualizam constant in care o sa se afle numarul cautat
            e - lista in care e[x] = True daca x este prim si False altfel
            numere_prime - lista in care se afla numerele prime pana la 1 milion fiindca nu se stie cat de mare poate fi n
    :return: variabila rezultat care conta in elementul pe care trebuie sa il aflam
    '''
    # indice = int(input("Numarul elementului pe care vreti sa il aflati: ")) # n-lea element
    if indice < 1:
        return "Nu exista"
    elemente = 1
    numar = rezultat = 1
    numere_prime = []
    e = [True for i in range(1000000)]
    eratostene(numere_prime, e)
    while(elemente < indice):
        if e[numar]:
            elemente += 1
            rezultat = numar
        else:
            index = 0
            # Aflarea divizorilor primi a numerelor compuse
            while(numere_prime[index]  <= numar // 2):
                if(numar % numere_prime[index] == 0):
                    rezultat = numere_prime[index]
                    elemente += 1
                    if elemente == indice :
                        return rezultat
                index += 1
       # print(elemente, " " , numar)
        numar += 1
    return rezultat

def teste_problema_12():
    assert (problema_12(1) == 1)
    assert (problema_12(2) == 2)
    assert (problema_12(3) == 3)
    assert (problema_12(4) == 2)
    assert (problema_12(5) == 5)
    assert (problema_12(6) == 2)
    assert (problema_12(7) == 3)
    assert (problema_12(8) == 7)
    assert (problema_12(20) == 5)

'''                       ///////////////////////                   '''

# Problema 2

def verifica_an_bisect(an):
    return True if (an % 4 == 0 and an % 100 != 0) or an % 400 == 0 else False

def nr_ani_bisecti(nastere, actual):
    numar = 0
    for i in range(nastere, actual):
        if verifica_an_bisect(i):
            numar += 1
    return numar

def rest_zile_an_nastere(luna, zi):
    zile_luni = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    nr_zile = zile_luni[luna] - zi
    for i in range(luna + 1, 13):
        nr_zile += zile_luni[i]
    return nr_zile

def rest_zile_an_actual(luna, zi):
    zile_luni = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    nr_zile = zi
    for i in range(1, luna):
        nr_zile += zile_luni[i]
    return nr_zile

def problema_2():
    ''' Problema se rezolva cu ajutorul librariei time care ne ajuta sa aflam anul, luna si ziua actuala '''
    from datetime import datetime
    '''      Date
        nr_total_zile - varsta persoanei in zile
    '''
    anul_actual = datetime.today().year
    luna_actuala = datetime.today().month
    ziua_actuala = datetime.today().day

    while True:
        try:
            an_nastere = int( input("Introduceti anul in care v-ati nascut: ") )
            if an_nastere > anul_actual:
                print("Valoare invalida.")
            else:
                break
        except ValueError:
            print("Valoare invalida.")

    while True:
        try:
            luna_nastere = int( input("Introduceti luna in care v-ati nascut sub forma de numar: ") )
            if an_nastere == anul_actual and luna_nastere > luna_actuala:
                print("Valoare invalida.")
            else:
                break
        except ValueError:
            print("Valoare invalida.")

    while True:
        try:
            zi_nastere = int( input("Introduceti ziua in care v-ati nascut: ") )
            if an_nastere == anul_actual and luna_nastere > luna_actuala and zi_nastere > ziua_actuala:
                print("Valoare invalida.")
            elif verifica_an_bisect(an_nastere) == False and luna_nastere == 2 and zi_nastere == 29:
                print("Valoare invalida.")
            else: break
        except ValueError:
            print("Valoare invalida.")

    nr_total_zile = (anul_actual - an_nastere - 1) * 365
    nr_total_zile += nr_ani_bisecti(an_nastere, anul_actual)
    nr_total_zile += rest_zile_an_nastere(luna_nastere, zi_nastere)
    nr_total_zile += rest_zile_an_actual(luna_actuala, ziua_actuala)
    if verifica_an_bisect(an_nastere) and luna_nastere > 2:
        nr_total_zile -= 1

    #print(nr_ani_bisecti(an_nastere, anul_actual))
    #print(rest_zile_an_nastere(luna_nastere, zi_nastere))
    #print(rest_zile_an_actual(luna_actuala, ziua_actuala))
    print(nr_total_zile)


'''                       ///////////////////////                   '''

if __name__ == "__main__":
    #problema_11()
    indice = int(input("Numarul elementului pe care vreti sa il aflati: ")) # n-lea element
    print(problema_12(1))
    # teste_problema_12()
    #for i in range(1, 42):
    #    print(problema_12(i), end = " ")
