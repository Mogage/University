# Proprietatile 11 si 12

def menu():
    """
        Se afiseaza meniul cu care interactioneaza utilizatorul
    :return: -
    """
    print("    Meniu: \n"
          " 1. Introduceti 1 pentru introducerea elementelor in lista. \n"
          " 2. Introduceti 2 pentru afisarea listei. \n"
          " 3. Introduceti 3 pentru scoaterea unui element din lista. \n"
          " 4. Introduceti 4 pentru gasirea secventei de lungime maxima care are suma maxima. \n"
          " 5. Introduceti 5 pentru gasirea secventei de lungime maxima care are oricare doua elemente consecutive de semne contrare. \n"
          " 6. Introduceti 6 pentru gasirea secventei de lungime maxima sub forma de munte. \n"
          " 7. Introduceti 7 pentru iesire."
          )

def remove_an_element_menu(list):
    """
        Se afiseaza meniul pentru stergerea unui element
    :return: -
    """
    list_length = len(list)
    exit_index = 2

    if list_length == 0:
        return 0

    print("    Indexarea elementelor se face incepand cu pozitia 1.\n"
          " 1. Introduceti 1 pentru stergerea primului element din lista.")
    if list_length >= 2:
        exit_index = 3
        print(" 2. Introduceti 2 pentru stergerea ultimului element din lista.")
    if list_length >= 3:
        exit_index = 4
        print(" 3. Introduceti 3 pentru stergerea unui element din lista.")
    print(" " + str(exit_index) + ". Introduceti", exit_index, "pentru iesire.")
    return exit_index

def read_list(list):
    """

    :return:
    """
    print("Introduceti un numar in lista sau exit daca doriti sa iesiti.")
    while True:
        number = input(">>>")
        if number.lower() == "exit":
            return
        else:
            try:
                number = int(number)
                list.append(number)
            except ValueError:
                print("Valoare introdusa invalida. Introduceti un numar in lista sau exit daca doriti sa iesiti.")

def print_list(list):
    """
        Se afiseaza elementele din lista
    :param list: list - lista in care utilizatorul a introdus valori
    :return: -
    """
    if len(list) == 0:
        print("Nu exista niciun element in lista.")
        return
    for index in list:
        print(index, " ", end=" ")
    print()

def remove_index_element(list):
    """
        Se elimina un element de pe o pozitie introdusa de utilizator, indexarea se face incepand cu 1
    :param list:
    :return: -
    """
    print("Introduceti indicele numarului pe care vreti sa il scoateti sau exit daca doriti sa iesiti: ")
    while True:
        index = input(">>>")
        if index.lower() == "exit":
            return
        else:
            try:
                index = int(index)
                list.pop(index - 1)
            except (ValueError, IndexError):
                print("Valoare introdusa invalida.")

def remove_an_element_main(list):
    """
        Se elimina un element din lista, indexarea se face incepand cu 1
    :param list: list - lista care urmeaza sa fie modificata in urma stergerii unui element
    :return: -
    """
    list_length = len(list)
    if list_length == 0:
        print("Nu exista niciun element in lista.")
        return
    while True:
        exit_index = remove_an_element_menu(list)
        if exit_index == 0:
            return

        user_input = input(">>>")

        if user_input == '1':
            list.pop(0)
        elif user_input == str(exit_index):
            return
        elif user_input == '2':
            list.pop(list_length - 1)
        elif user_input == '3':
            remove_index_element(list)
        else:
            print("Valoarea introdusa este invalida.")

def property_1_sequence(list):
    """
        Gaseste o secventa de lungime maxima care are suma maxima
        In cazul in care exista mai multe se ia cea mai din stanga
        variabile int:
            max_sum - suma maxima
            auxiliary_index - indice cu care ar putea incepe o secventa
            left_index - pozitia de pe care incepe secventa cautata
            right_index - pozitia pe care se termina secventa cautata
            list_index - indice pentru parcurgerea listei
            list_length - indicele pana la care parcugem lista (e lungimea acesteia fiindca o parcurgem pe toata)
            auxiliary_sum - var pentru calcularea sumelor intermediare
            sequence_max_length - lungimea maxima a secventei cautate
    :param list:
    :return: -
    """
    if len(list) == 0:
        print("Nu exista niciun element in lista.")
        return
    # Var
    max_sum = list[0]
    auxiliary_index = 0
    auxiliary_sum = list[0]
    left_index = 0
    right_index = 0
    list_index = 1
    list_length = len(list)
    sequence_max_length = 1
    sequence_length = 1

    # Parcurgerea listei si cautarea secventei cu proprietatea ceruta
    while list_index < list_length:
        if auxiliary_sum < 0:                       # o secventa de suma maxima nu va incepe niciodata cu un numar negativ,
            auxiliary_sum = list[list_index]        # iar in cazul in care se ajunge la o suma negativa, o actualizam cu elementul curent
            auxiliary_index = list_index            # care reprezinta solutia optima la pasul curent
            sequence_length = 1
        else:                                       # In caz contrar adaugam la suma elementul curent
            auxiliary_sum += list[list_index]
            sequence_length += 1
        if auxiliary_sum > max_sum or auxiliary_sum == max_sum and sequence_length > sequence_max_length:
            max_sum = auxiliary_sum                  # Daca suma secventei este mai mare decat cea maxima actualizam corespunzator variabilele de iesire
            right_index = list_index
            left_index = auxiliary_index
            sequence_max_length = sequence_length
        list_index += 1

    # Afisarea rezultatului
    print("Secventa de lungime", sequence_max_length, "care are suma maxima", max_sum, "este: ", end="")
    for i in range(left_index, right_index + 1):    # Afisarea secventei cautate
        print(list[i], end=" ")
    print()

    return max_sum  # In caz de teste

def get_sign(number):
    """
    :param number: number - parametru de tip int
    :return: True daca number < 0, False alfel
    """
    return True if number < 0 else False

def property_2_sequence(list):
    """
        Gaseste o secventa de lungime maxima care are oricare doua elemente consecutive de semne contrare.
        In cazul in care exista mai multe se ia cea mai din stanga
        variabile bool:
            sign - semnul pe care l-a avut elementul precedent, True daca e negativ, False altfel
        variabile int:
            left_index - pozitia de pe care incepe secventa cautata
            right_index - pozitia pe care se termina secventa cautata
            auxiliary_index -  indice cu care ar putea incepe o secventa
            sequence_length - lungimea maxima a secventei cautate
            auxiliary_length - var pentru calcularea lungimilor intermediare
            list_index - indice pentru parcurgerea listei
            list_length - indicele pana la care parcugem lista (e lungimea acesteia fiindca o parcurgem pe toata)
    :param list:
    :return:
    """
    if len(list) == 0:
        print("Nu exista niciun element in lista.")
        return
    # Var
    if list[0] < 0:
        sign = True
    else:
        sign = False
    left_index = 0
    right_index = 0
    auxiliary_index = 0
    sequence_length = 1
    auxiliary_length = 1
    list_index = 1
    list_length = len(list)

    # Parcurgerea listei si cautarea secventei cu proprietatea ceruta
    while list_index < list_length:
        if sign == get_sign(list[list_index]):      # Daca elementul curent are acelasi semn ca cel anterior nu se respecta proprietatea
            auxiliary_index = list_index            # si presupunem ca de la pozitia curenta se poate incepe un nou sir
            auxiliary_length = 1
        else:                                       # Altfel, se respecta proprietatea si incrementam lungimea curenta a sirului
            auxiliary_length += 1
            sign = not sign
        if auxiliary_length > sequence_length:      # Daca lungimea sirului din momentul respectiv este mai mare decat cea maxima
            left_index = auxiliary_index            # actualizam variabilele de iesire corespunzator
            right_index = list_index
            sequence_length = auxiliary_length
        list_index += 1

    # Afisarea rezultatului
    print("Lungimea maxima a secventei care are oricare doua elemente consecutive de semne contrare este", sequence_length,
          "\nSecventa corespunzatoare este:", end=" ")
    for sequence_index in range(left_index, right_index + 1):   # Afisarea secventei cautate
        print(list[sequence_index], end=" ")
    print()

    return sequence_length  # In caz de teste

def ascend(list, list_index):
    """

    :param list:
    :param list_index:
    :return:
    """
    list_length = len(list)
    while list_index < list_length - 1 and list[list_index] < list[list_index + 1]:
        list_index += 1
    return list_index

def descend(list, list_index):
    """

    :param list:
    :param list_index:
    :return:
    """
    list_length = len(list)
    while list_index < list_length - 1 and list[list_index] > list[list_index + 1]:
        list_index += 1
    return list_index

def property_3_sequence(list):
    """
        Cauta secventa de lungime maxima de tip munte
        In cazul in care exista mai multe secvente cu aceleasi lungimi se ia cea mai din stanga
        variabile int:
            left_index - pozitia de pe care incepe secventa cautata
            max_length - lungimea maxima a secventei cautate
            list_index - indice pentru parcurgerea listei
            list_length - indicele pana la care parcugem lista (e lungimea acesteia fiindca o parcurgem pe toata)
            aux_list - lista in care se va afla secventa cautata
            top_position - pozitia pe care se afla un varf de munte
    :param list: list -
    :return: secventa de lungime maxima sub forma de munte, daca exista, alfel mesajul 'Nu exista.'
    """
    if len(list) == 0:
        print("Nu exista niciun element in lista.")
        return

    # Var
    list_index = 0
    list_length = len(list)
    max_length = 0
    aux_list = []
    top_position = 0

    list_index = descend(list, list_index)                                          # Trecem peste primii termeni descrescatori
    while list_index < list_length:
        left_index = list_index
        list_index = ascend(list, list_index)
        if list_index < list_length:
            top_position = list_index
        list_index = descend(list, list_index)

        if top_position != list_index and list_index - left_index + 1 > max_length:
            max_length = list_index - left_index + 1
            aux_list = list[left_index:list_index + 1]
        if list_index < list_length - 1 and list[list_index] < list[list_index + 1]:
            list_index -= 1
        list_index += 1

    if len(aux_list) > 0:
        return aux_list
    return "Nu exista."

def test_property_1_sequence():
    assert (property_1_sequence([0, 0, 0, 0]) == 0)
    assert (property_1_sequence([-2, -5, -7, -1, -10]) == -1)
    assert (property_1_sequence([-4, 1, -5, 1, 4, -2, 2, 3, -4, 4]) == 8)
    assert (property_1_sequence([-2, -5, -7, -1, 0]) == 0)
    assert (property_1_sequence([2, 5, 4, -100, 1, 2, 4, 4]) == 11)
    assert (property_1_sequence([-2, -5, 0, 0, 1, 0, -2]) == 1)

def test_property_2_sequence():
    assert property_2_sequence([5, -2, 0, 5, 6, -4, 8, -2]) == [4, [6, -4, 8, -2]]
    assert property_2_sequence([2, 0, 5, 4, 3]) == [1, [2]]
    assert property_2_sequence([-2, -6, -7, -10, -1]) == [1, [-2]]
    assert property_2_sequence([1, -1, 1, -1, 1, -1]) == [6, [1, -1, 1, -1, 1, -1]]
    assert property_2_sequence([-1, 1, -1, 1, -1, 1]) == [6, [-1, 1, -1, 1, -1, 1]]
    assert property_2_sequence([0, 0, 0]) == [1, [0]]

def test_property_3_sequence():
    assert property_3_sequence([1, 2, 3, 5]) == "Nu exista."
    assert property_3_sequence([1, 2, 6, 5]) == [1, 2, 6, 5]
    assert property_3_sequence([9, 8, 7, 2, 3]) == "Nu exista."
    assert property_3_sequence([1, 5, 3, 10, 11, 4, 7, 5]) == [3, 10, 11, 4]
    assert property_3_sequence([2, 5, 5]) == "Nu exista."
    assert property_3_sequence([7, 8, 7, 2, 3, 6]) == [7, 8, 7, 2]
    assert property_3_sequence([2, 3, 5, 4, 7, 8, 6]) == [2, 3, 5, 4]

def test_get_sign():
    assert get_sign(0) == False
    assert get_sign(5) == False
    assert get_sign(-2) == True

def main_program():
    """
        Programul principal in care se fac operatiile
    :return:
    """
    print("                *Aplicatie laborator 3*")
    list = []
    while True:
        menu()

        user_input = input(">>>")

        if user_input == '':
            continue

        if user_input == '1':
            read_list(list)
        elif user_input == '2':
            print_list(list)
        elif user_input == '3':
            remove_an_element_main(list)
        elif user_input == '4':
            property_1_sequence(list)
        elif user_input == '5':
            property_2_sequence(list)
        elif user_input == '6':
            print(property_3_sequence(list))
        elif user_input == '7':
            break
        else:
            print("Valoarea introdusa este invalida. Introduceti o valoare din meniu.")

    print("La revedere!")

if __name__ == "__main__":
    # test_property_1_sequence()
    # test_property_2_sequence()
    # test_get_sign()
    test_property_3_sequence()
    main_program()
