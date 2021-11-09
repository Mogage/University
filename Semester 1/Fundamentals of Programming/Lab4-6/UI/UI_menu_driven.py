import service
import undo
from expenses import check_type, to_str_expense, get_expense_type

def ui_menu():
    """
        Afiseaza meniul cu care interactioneaza utilizatorul
    :return: -
    """
    print("    Meniu:\n"   
          " 1. Introduceti 1 pentru a adauga o noua cheltuiala.\n"
          " 2. Introduceti 2 pentru a actualiza o cheltuiala.\n"
          " 3. Introduceti 3 pentru a sterge cheltuielile dintr-o zi.\n"
          " 4. Introduceti 4 pentru a sterge cheltuielile dintr-un interval de timp.\n"
          " 5. Introduceti 5 pentru a sterge cheltuielile de un anumit tip.\n"
          " 6. Introduceti 6 pentru a tipari toate cheltuielile.\n"
          " 7. Introduceti 7 pentru a tipari cheltuielile mai mari decat o suma.\n"
          " 8. Introduceti 8 pentru a tipari cheltuielile efectuate inainte de o zi si mai mici decat o suma.\n"
          " 9. Introduceti 9 pentru a tipari toate cheltuielile de un anumit tip.\n"
          " 10. Introduceti 10 pentru a tipari suma totala pentru un anumit tip de cheltuiala.\n"
          " 11. Introduceti 11 pentru a tipari ziua in care suma cheltuita e maxima.\n"
          " 12. Introduceti 12 pentru a tipari cheltuielile ce au o anumita suma.\n"
          " 13. Introduceti 13 pentru a tipari cheltuielile sortate dupa tip.\n"
          " 14. Introduceti 14 pentru a filtra cheltuielile de un anumit tip.\n"
          " 15. Introduceti 15 pentru a filtra cheltuielile mai mici decat o suma.\n"
          " 16. Introduceti 16 pentru a anula ultima operatie.\n"
          " 17. Introduceti 17 pentru a iesi din program.")

def ui_add_expense(list):
    """
        Se citesc datele cheltuielii si se adauga cheltuiala la lista list
    :param list: lista de cheltuieli
    :return: -
    """
    try:
        print("Introduceti ziua pentru care ati efectuat cheltuiala.")
        day = int(input(">>>"))
    except ValueError:
        print("Ziua introdusa este invalida.")
        return
    print("Introduceti tipul cheltuielii.")
    type = input(">>>")
    try:
        print("Introduceti suma cu care ati efectuat cheltuiala.")
        value = int(input(">>>"))
    except ValueError:
        print("Suma introdusa este invalida.")
        return

    service.srv_add_expense_to_list(list, day, value, type)

    print("Cheltuiala introdusa cu succes!")

def ui_update_expense(list, undo_stack):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala si intreaba
        utilizatorul daca doreste sa adauge una noua, in caz afirmativ continua cu adaugarea unei noi cheltuieli altfel de iese din functie
        Daca exista cheltuieli se citeste cheltuiala care sa fie actualizata si noua suma si se actualizeaza daca datele sunt valide
    :param list: lista de cheltuieli
    :param undo_stack: stiva cu ultimele operatii efectuate
    :return:
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return
    try:
        print("Introduceti ziua pentru care vreti sa actualizati cheltuiala.")
        day = int(input(">>>"))
    except ValueError:
        print("Ziua introdusa este invalida.")
        return
    print("Introduceti tipul cheltuielii pe care vreti sa o actualizati.")
    type = input(">>>")
    try:
        print("Introduceti noua suma.")
        value = int(input(">>>"))
    except ValueError:
        print("Suma introdusa este invalida.")
        return

    service.srv_update_expense(list, day, value, type, undo_stack)

def ui_delete_expense_by_day(list, undo_stack):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel se citeste ziua cheltuielilor care sa se stearga si le sterge daca ziua este valida
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :param undo_stack: stiva cu ultimele operatii efectuate
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti ziua cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input = input(">>>")

        if user_input == "":
            continue

        if "1" <= user_input <= "31":
            print(service.srv_delete_expense_by_day(list, int(user_input), undo_stack))
            return
        elif user_input == "exit":
            return
        else:
            print("Zi introdusa invalida.")
        print("Introduceti ziua cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")

def ui_delete_expense_between_days(list, undo_stack):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel se citesc zilele intervalului din care se vor sterge cheltuieli si daca sunt valide se continua cu stergerea
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :param undo_stack: stiva cu ultimele operatii efectuate
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti ziua de inceput a cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input_1 = input(">>>")

        if user_input_1 == "":
            continue

        if "1" <= user_input_1 <= "31":
            print("Introduceti ziua de sfarsit a cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")
            while True:
                user_input_2 = input(">>>")

                if user_input_2 == "":
                    continue

                if user_input_2 >= user_input_1 and "1" <= user_input_2 <= "31":
                    print(service.srv_delete_expense_between_days(list, int(user_input_1), int(user_input_2), undo_stack))
                    return
                elif user_input_2 == "exit":
                    return
                else:
                    print("Zi introdusa invalida.")
                print(
                    "Introduceti ziua de sfarsit a cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")
            return
        elif user_input_1 == "exit":
            return
        else:
            print("Zi introdusa invalida.")
        print("Introduceti ziua de inceput a cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")

def ui_delete_expense_by_type(list, undo_stack):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel se citeste tipul cheltuielilor care sa se stearga si le sterge daca tipul este valid
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :param undo_stack: stiva cu ultimele operatii efectuate
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti tipul cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input = input(">>>")

        if user_input == "":
            continue

        if check_type(user_input):
            print(service.srv_delete_expense_by_type(list, user_input, undo_stack))
            return
        elif user_input == "exit":
            return
        else:
            print("Tip introdus invalid.")
        print("Introduceti tipul cheltuielilor pe care vreti sa le stergeti sau exit ca sa va intoarceti la meniu.")

def ui_print_expenses(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel tipareste elementele din lista in formatul prestabilit
    :param list: lista de cheltuieli
    :return:
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print(service.srv_print_expenses(list))

def ui_print_expense_greater_than_value(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel citeste de la utilizator suma peste care sa se tipareasca cheltuielile si daca este valida se continua cu tiparirea
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti suma peste care doriti sa se tipareasca cheltuielile sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input = input(">>>")

        if user_input == "":
            continue

        if "1" <= user_input <= "9999":
            print(service.srv_print_expense_greater_than_value(list, int(user_input)))
            return
        elif user_input == "exit":
            return
        else:
            print("Suma introdusa invalida.")
        print("Introduceti suma peste care doriti sa se tipareasca cheltuielile sau exit ca sa va intoarceti la meniu.")

def ui_print_expense_before_day_lower_value(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel se citeste ziua inaintea carui sa se caute cheltuielile care au suma mai mica decat o suma citita de la utilizator
        sau comanda exit daca se vrea intoarecerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti ziua inaintea caruia sa se tipareasca cheltuielile sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input_day = input(">>>")

        if user_input_day == "":
            continue

        if "1" <= user_input_day <= "31":
            print("Introduceti suma sub care doriti sa se tipareasca cheltuielile sau exit ca sa va intoarceti la meniu.")
            while True:
                user_input_value = input(">>>")

                if user_input_value == "":
                    continue

                if "1" <= user_input_value <= "9999":
                    print(service.srv_print_expense_before_day_lower_value(list, int(user_input_day), int(user_input_value)))
                    return
                elif user_input_day == "exit":
                    return
                else:
                    print("Ziua introdusa invalida.")
                print("Introduceti suma sub care doriti sa se tipareasca cheltuielile sau exit ca sa va intoarceti la meniu.")
            return
        elif user_input_day == "exit":
            return
        else:
            print("Ziua introdusa invalida.")
        print("Introduceti ziua inaintea caruia sa se tipareasca cheltuielile sau exit ca sa va intoarceti la meniu.")

def ui_print_expense_by_type(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel se citeste tipul cheltuielilor care sa se tipareasca si le tipareste daca tipul este valid
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    while True:
        print("Introduceti tipul cheltuielilor pe care vreti sa le tipariti sau exit ca sa va intoarceti la meniu.")

        user_input = input(">>>")

        if user_input == "":
            continue

        if check_type(user_input):
            print(service.srv_print_expense_by_type(list, user_input))
            return
        elif user_input == "exit":
            return
        else:
            print("Tip introdus invalid.")

def ui_print_total_values_by_type(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel se citeste tipul cheltuielii pentru care se vrea suma totala si o calculeaza daca tipul este valid
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    while True:
        print("Introduceti tipul pentru care vreti sa aflati suma totala sau exit ca sa va intoarceti la meniu.")

        user_input = input(">>>")

        if user_input == "":
            continue

        if check_type(user_input):
            print(service.srv_print_total_values_by_type(list, user_input))
            return
        elif user_input == "exit":
            return
        else:
            print("Tip introdus invalid.")

def ui_find_max_value_expense(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel, cauta ziua in care s-a cheltuit cea mai mare suma si o tipareste sub forma:
        "In ziua " + ziua_cautata + " s-a cheltuit cea mai mare suma de" + suma_cautata +
        + " leu.", daca suma_cautata = 1
        + " lei.", daca suma_cautata > 1 and < 20
        + " de lei.", daca suma_cautata >= 20
    :param list: lista de cheltuieli
    :return: -

    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    day_sum = service.srv_find_max_value_expense(list)

    print("In ziua", day_sum[0], "s-a cheltuit cea mai mare suma de", day_sum[1], end=" ")

    if day_sum[1] == 1:
        print("leu.")
    elif day_sum[1] < 20:
        print("lei.")
    else:
        print("de lei.")

def ui_print_expense_by_value(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel citeste de la utilizator suma cheltuielilor care sa fie afisate si daca e valida se continua cu tiparirea
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    while True:
        print("Introduceti suma cheltuielilor pe care vreti sa le tipariti sau exit ca sa va intoarceti la meniu.")

        user_input = input(">>>")

        if user_input == "":
            continue

        if "1" <= user_input <= "9999":
            print(service.srv_print_expense_by_value(list, int(user_input)))
            return
        elif user_input == "exit":
            return
        else:
            print("Suma introdusa invalida.")

def ui_print_expense_ordered_by_type(list):
    """
        Tipareste o lista sortata alfabetic dupa tip
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    ordered_list = sorted(list, key=get_expense_type)

    for expense in ordered_list:
        print(to_str_expense(expense))

def ui_expense_filter_by_type(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel citeste de la utilizator tipul cheltuielilor care sa fie filtrate si daca e valid se continua cu tiparirea
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti tipul cheltuielilor pe care vreti sa le flitrati sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input = input(">>>")

        if user_input == "":
            continue

        if check_type(user_input):
            filtered_list = service.srv_expense_filter_by_type(list, user_input)
            if service.srv_empty_list(filtered_list):
                print("Nu exista alte cheltuieli cu tip diferit.")
            else:
                for expense in filtered_list:
                    print(to_str_expense(expense))
            return
        elif user_input == "exit":
            return
        else:
            print("Tip introdus invalid.")
        print("Introduceti tipul cheltuielilor pe care vreti sa le flitrati sau exit ca sa va intoarceti la meniu.")

def ui_expense_filter_lower_value(list):
    """
        Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
        Altfel citeste de la utilizator suma sub care sa fie filtrate cheltuielile si daca e valida se continua cu tiparirea
        sau comanda exit daca se vrea intoarcerea la meniu
    :param list: lista de cheltuieli
    :return: -
    """
    if service.srv_empty_list(list):
        print("Nu exista nicio cheltuiala inregistrata.")
        return

    print("Introduceti suma sub care vreti sa filtrati cheltuielile sau exit ca sa va intoarceti la meniu.")
    while True:
        user_input = input(">>>")

        if user_input == "":
            continue

        if "1" <= user_input <= "9999":
            filtered_list = service.srv_expense_filter_lower_value(list, int(user_input))
            if service.srv_empty_list(filtered_list):
                print("Nu exista cheltuieli mai mari decat aceasta suma.")
            else:
                for expense in filtered_list:
                    print(to_str_expense(expense))
            return
        elif user_input == "exit":
            return
        else:
            print("Suma introdusa este invalida.")
        print("Introduceti suma sub care vreti sa filtrati cheltuielile sau exit ca sa va intoarceti la meniu.")

def ui_undo(list, stack):
    """
        Anuleaza ultima operatie care s-a efectuat
    :param list: lista de cheltuieli
    :param stack: stiva cu ultimele operatii efectuate
    :return: -
    """
    if service.srv_empty_list(stack):
        print("Nu a fost facuta nicio operatie.")
        return

    top = stack[-1]

    if undo.get_code(top) == 1:
        service.srv_remove_last_element(list)
    elif undo.get_code(top) == 2:
        undo.update(list, top)
    elif undo.get_code(top) == 3:
        undo.delete(list, top)

    print("Operatia a fost anulata.")

    service.srv_remove_last_element(stack)

def ui_user():
    """
        Functie prin care se interactioneaza cu utilizatorul, aici se afiseaza meniul si se introduce optiunea utilizatorului
        1 - Adauga cheltuiala
        2 - Actualizeaza cheltuiala
        3 - Se sterg toate cheltuielile pentru o zi data
        4 - Se sterg cheltuielile dintr-un interval de timp
        5 - Se sterg cheltuielile de un tip
        6 - Se tiparesc toate cheltuielile
        7 - Se tiparesc cheltuielile mai mari decat o suma
        8 - Se tiparesc cheltuielile efectuate inainte de o zi si mai mici decat o suma
        9 - Se tiparesc cheltuielile de un anumit tip
        10 - Se tipareste suma totala pentru un anumit tip de cheltuiala
        11 - Se gaseste ziua in care suma cheltuita e maxima
        12 - Se tiparesc toate cheltuielile ce au o anumita suma
        13 - Se tiparesc cheltuielile sortate dupa tip
        14 - Filtreaza / Elimina cheltuielile de un anumit tip
        15 - Filtreaza / Elimina cheltuielile mai mici decat o suma data
        16 - Anuleaza ultima operatie
        17 - Se iese din program
    :return: -
    """
    list = []
    undo_stack = []
    print("                 **Aplicatie laborator 4-6**")
    ui_menu()
    while True:
        user_input = input(">>>")

        if user_input == "":
            continue

        if user_input == "1":
            try:
                ui_add_expense(list)
                undo_stack.append({"code": 1})
            except Exception as ex:
                print(ex, end="")
        elif user_input == "2":
            try:
                ui_update_expense(list, undo_stack)
            except Exception as ex:
                print(ex, end="")
        elif user_input == "3":
            ui_delete_expense_by_day(list, undo_stack)
        elif user_input == "4":
            ui_delete_expense_between_days(list, undo_stack)
        elif user_input == "5":
            ui_delete_expense_by_type(list, undo_stack)
        elif user_input == "6":
            ui_print_expenses(list)
        elif user_input == "7":
            ui_print_expense_greater_than_value(list)
        elif user_input == "8":
            ui_print_expense_before_day_lower_value(list)
        elif user_input == "9":
            ui_print_expense_by_type(list)
        elif user_input == "10":
            ui_print_total_values_by_type(list)
        elif user_input == "11":
            ui_find_max_value_expense(list)
        elif user_input == "12":
            ui_print_expense_by_value(list)
        elif user_input == "13":
            ui_print_expense_ordered_by_type(list)
        elif user_input == "14":
            ui_expense_filter_by_type(list)
        elif user_input == "15":
            ui_expense_filter_lower_value(list)
        elif user_input == "16":
            ui_undo(list, undo_stack)
        elif user_input == "17":
            return
        else:
            print("Valoarea introdusa este invalida. Trebuie sa fie un numar din meniu.")

        ui_menu()
