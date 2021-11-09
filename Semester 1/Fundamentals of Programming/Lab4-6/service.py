import expenses

def srv_remove_last_element(list):
    """
        Sterge ultimul Cheltuiala al unei liste
    :param list: o lista
    :return: -
    """
    list.pop()

def srv_empty_list(list):
    """
        Verifica daca o lista este goala
    :param list: o lista
    :return: True daca lungimea listei = 0, False altfel
    """
    if len(list) == 0:
        return True
    return False

def srv_add_expense_to_list(list, day, value, type):
    """
        Creeaza o cheltuiala cu ziua day, suma value si tipul type, verifica daca e valid aceasta cheltuiala iar in caz afirmativ
        il adauga la lista list de cheltuieli unic identificabile doar daca nu exista deja o cheltuiala de acelasi tip in aceeasi zi
    :param list: lista de cheltuieli
    :param day: int >= 1 and <= 31
    :param value: int >= 1 and <= 9999
    :param type: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :return: -
    """
    expense = expenses.create_expense(day, value, type)
    expenses.check_expense(expense)
    expenses.add_expense_to_list(list, expense)

def srv_update_expense(list, day, value, type, undo_stack=None):
    """
        Actualizeaza o cheltuiala de tip type din ziua day cu suma value, daca aceasta exista
        si adauga in undo_stack elementul inainte de modificare alaturi de pozitia pe care se afla si codul de operatie 2
    :param list: lista de cheltuieli
    :param undo_stack: stiva cu ultimele operatii efectuate
    :param day: int >= 1 and <= 31
    :param value: int >= 1 and <= 9999
    :param type: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :return: -
    :raise Exception cu textul:
            "Elementul nu exista", daca nu exista o cheltuiala de acelasi tip in aceeasi zi
    """
    if undo_stack is None:
        undo_stack = []
    updated_expense = expenses.create_expense(day, value, type)
    expenses.check_expense(updated_expense)
    list_index = 0
    list_length = len(list)
    while list_index < list_length:
        if expenses.equal_expenses(list[list_index], updated_expense):
            undo_stack.append({"code": 2, "expense": list[list_index], "position": list_index})
            list[list_index] = updated_expense
            return
        list_index += 1

    raise Exception("Cheltuiala nu exista.\n")

def srv_delete_expense_by_day(list, day, undo_stack=None):
    """
        Sterge cheltuielile dintr-o zi day si adauga in undo_stack elementele sterse alaturi de pozitiile lor
        si codul de operatie 3
    :param list: lista de cheltuieli
    :param undo_stack: stiva cu ultimele operatii efectuate
    :param day: int din intervalul [1, 31]
    :return: Mesajul "Cheltuieli sterse cu succes.", daca au fost sterse mai mult de o cheltuiala
             Mesajul "Cheltuiala stearsa cu succes.", daca a fost stearsa doar o cheltuiala
             Mesajul "Nu s-a efectuat nicio cheltuiala de acest tip", altfel
    """
    if undo_stack is None:
        undo_stack = []
    list_length = len(list)
    list_index = 0
    deleted_expenses = []
    while list_index < list_length:
        if 0 <= list_index < list_length and expenses.get_expense_day(list[list_index]) == day:
            deleted_expenses.append({"expense": list[list_index], "position": list_index})
            list.pop(list_index)
            list_index -= 1
            list_length -= 1
        list_index += 1

    numbers_of_elements_deleted = len(deleted_expenses)

    if numbers_of_elements_deleted > 1:
        undo_stack.append({"code": 3, "deleted_expenses": deleted_expenses})
        return "Cheltuieli sterse cu succes."

    if numbers_of_elements_deleted == 1:
        undo_stack.append({"code": 3, "deleted_expenses": deleted_expenses})
        return "Cheltuiala stearsa cu succes."

    return "Nu s-a efectuat nicio cheltuiala in aceasta zi."

def srv_delete_expense_between_days(list, day_1, day_2, undo_stack=None):
    """
        Sterge cheltuielele dintr-un interval de timp [day_1, day_2] si adauga in undo_stack elementele sterse alaturi de pozitiile lor
        si codul de operatie 3
    :param list: lista de cheltuieli
    :param day_1: int din intervalul [1, 31], prima zi din interval
    :param day_2: int din intervalul [1, 31], a doua zi din interval, day_2 >= day_1
    :param undo_stack: stiva cu ultimele operatii efectuate
    :return: Mesajul "Cheltuieli sterse cu succes", daca se gasesc elemente si se sters
             Mesajul "Nu s-a efectuat nicio cheltuiala in acest interval de timp", altfel
    """
    if undo_stack is None:
        undo_stack = []
    list_length = len(list)
    list_index = 0
    deleted_expenses = []
    while list_index < list_length:
        if 0 <= list_index < list_length and day_1 <= expenses.get_expense_day(list[list_index]) <= day_2:
            deleted_expenses.append({"expense": list[list_index], "position": list_index})
            list.pop(list_index)
            list_index -= 1
            list_length -= 1
        list_index += 1

    numbers_of_elements_deleted = len(deleted_expenses)

    if numbers_of_elements_deleted >= 1:
        undo_stack.append({"code": 3, "deleted_expenses": deleted_expenses})
        return "Cheltuieli sterse cu succes."

    return "Nu s-a efectuat nicio cheltuiala in acest interval de timp."

def srv_delete_expense_by_type(list, type, undo_stack=None):
    """
        Sterge cheltuielile de tip type din lista list si adauga in undo_stack elementele sterse alaturi de pozitiile lor
        si codul de operatie 3
    :param list: lista de cheltuieli
    :param type: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :param undo_stack: stiva cu ultimele operatii efectuate
    :return: Mesajul "Cheltuieli sterse cu succes.", daca au fost sterse mai mult de o cheltuiala
             Mesajul "Cheltuiala stearsa cu succes.", daca a fost stearsa doar o cheltuiala
             Mesajul "Nu s-a efectuat nicio cheltuiala de acest tip", altfel
    """
    if undo_stack is None:
        undo_stack = []
    list_length = len(list)
    list_index = 0
    deleted_expenses = []
    while list_index < list_length:
        if 0 <= list_index < list_length and expenses.get_expense_type(list[list_index]) == type:
            deleted_expenses.append({"expense": list[list_index], "position": list_index})
            list.pop(list_index)
            list_index -= 1
            list_length -= 1
        list_index += 1

    numbers_of_elements_deleted = len(deleted_expenses)

    if numbers_of_elements_deleted > 1:
        undo_stack.append({"code": 3, "deleted_expenses": deleted_expenses})
        return "Cheltuieli sterse cu succes."

    if numbers_of_elements_deleted == 1:
        undo_stack.append({"code": 3, "deleted_expenses": deleted_expenses})
        return "Cheltuiala stearsa cu succes."

    return "Nu s-a efectuat nicio cheltuiala de acest tip."

def srv_print_expenses(list):
    """

    :param list: lista de cheltuieli
    :return: -
    """
    expenses_string = ""
    for expense in list:
        expenses_string += expenses.to_str_expense(expense)
        expenses_string += "\n"

    expenses_string = expenses_string[0:-1]

    return expenses_string

def srv_print_expense_greater_than_value(list, value):
    """
        Cauta cheltuielile care au suma value si le adauga intr-un string pentru a fi afisate pe linii separate
    :param list: lista de cheltuieli
    :param value: int din intervalul [1, 9999]
    :return: Mesajul "Nu s-a efectuat nicio cheltuiala mai mare decat aceasta suma.", daca nu exista cheltuieli cu sume mai mari decat value
             Un string ce contine cheltuielile care au suma mai mare decat value pe linii separate, altfel
    """
    operations = ""
    for expense in list:
        if expenses.get_expense_value(expense) > value:
            operations += expenses.to_str_expense(expense)
            operations += "\n"

    operations = operations[0:-1]

    if operations == "":
        return "Nu s-a efectuat nicio cheltuiala mai mare decat aceasta suma."

    return operations

def srv_print_expense_before_day_lower_value(list, day, value):
    """
        Cauta cheltuielile care s-au efectuat inainte de ziua day si care au suma mai mica decat suma value
        si le adauga intr-un string pentru a fi afisate pe linii separate
    :param list: lista de cheltuieli
    :param day: int din intervalul [1, 31]
    :param value: int din intervalul [1. 9999]
    :return: Mesajul "Nu s-a efectuat nicio cheltuiala inainte de aceasta zi.", daca nu sunt cheltuieli efectuate inainte de ziua day
             Mesajul "Nu s-a efectuat nicio cheltuiala mai mica decat aceasta suma.", daca sunt cheltuielie efectuate inainte de ziua day dar nu au suma mai mica decat suma value
             Un string ce contine cheltuielile care s-au efectuat inainte de ziua day si au suma mai mica decat suma value pe linii separate, altfel
    """
    operations = ""
    valid_day = False
    for expense in list:
        if expenses.get_expense_day(expense) < day:
            valid_day = True
            if expenses.get_expense_value(expense) < value:
                operations += expenses.to_str_expense(expense)
                operations += "\n"

    operations = operations[0:-1]

    if not valid_day:
        return "Nu s-a efectuat nicio cheltuiala inainte de aceasta zi."

    if operations == "":
        return "Nu s-a efectuat nicio cheltuiala mai mica decat aceasta suma."

    return operations

def srv_print_expense_by_type(list, type):
    """
        Adauga toate cheltuielile de tip type efectuate intr-un string pentru a putea fi afisate fiecare pe linie separata
    :param list: o lista de cheltuieli
    :param type: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :return: Mesajul "Nu s-a efectuat nicio cheltuiala pentru acest tip.", daca nu exista nicio cheltuiala de tip type
             Un string ce contine cheltuielile de tip type efectuate, altfel
    """
    operations = ""
    for expense in list:
        if expenses.get_expense_type(expense) == type:
            operations += expenses.to_str_expense(expense)
            operations += "\n"

    operations = operations[0:-1]

    if operations == "":
        return "Nu s-a efectuat nicio cheltuiala de acest tip."

    return operations

def srv_print_total_values_by_type(list, type):
    """
        Calculeaza suma totala pentru un anumit tip type de cheltuiala
    :param list: o lista de cheltuieli
    :param type: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :return: Mesajul "Nu s-a efectuat nicio cheltuiala de acest tip", daca nu exista nicio cheltuiala de tip type
            Mesajul "Suma totala pentru tipul de cheltuieli selectat este de " + suma totala calculata +
            + " leu.", daca suma totala = 1
            + " lei.", daca suma totala > 1 and < 20
            + " de lei.", daca suma totala >= 20
    """
    total_sum = 0
    for expense in list:
        if expenses.get_expense_type(expense) == type:
            total_sum += expenses.get_expense_value(expense)

    if total_sum == 0:
        return "Nu s-a efectuat nicio cheltuiala de acest tip."

    output = "Suma totala pentru tipul de cheltuieli selectat este de " + str(total_sum)

    if total_sum == 1:
        output += " leu."
    elif total_sum < 20:
        output += " lei."
    else:
        output += " de lei."

    return output

def srv_find_max_value_expense(list):
    """
        Cauta ziua in care suma cheltuita este maxima
    :param list: o lista de cheltuieli
    :return: lista de tipul [zi, suma_maxima]
    """
    days = {}
    for expense in list:
        day = expenses.get_expense_day(expense)
        value = expenses.get_expense_value(expense)
        if day in days:
            days[day] += value
        else:
            days[day] = value

    sum_max = 0
    result = 0
    for key, value in days.items():
        if value > sum_max:
            sum_max = value
            result = key

    return [result, sum_max]

def srv_print_expense_by_value(list, value):
    """
        Cauta cheltuielile care au suma value si le adauga intr-un string pentru a se afisa pe linii separate
    :param list: lista de cheltuieli
    :param value: int din intervalul [1, 9999]
    :return: Mesajul "Nu s-a efectuat nicio cheltuiala cu aceasta suma.", daca nu exista cheltuieli cu suma value
             Un string ce contine cheltuielile cu suma value efectuate, altfel
    """
    operations = ""
    for expense in list:
        if expenses.get_expense_value(expense) == value:
            operations += expenses.to_str_expense(expense)
            operations += "\n"

    operations = operations[0:-1]

    if operations == "":
        return "Nu s-a efectuat nicio cheltuiala cu aceasta suma."

    return operations

def srv_expense_filter_by_type(list, type):
    """
        Cauta cheltuielile de un tip diferit de type si le adauga intr-o lista
    :param list: lista de cheltuieli
    :param type: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :return: lista filtrata
    """
    filtered_list = []
    for expense in list:
        if expenses.get_expense_type(expense) != type:
            filtered_list.append(expense)

    return filtered_list

def srv_expense_filter_lower_value(list, value):
    """
        Cauta cheltuielile cu o suma mai mare decat value si le adauga intr-o lista
    :param list: lista de cheltuieli
    :param value: int din intervalul [1, 9999]
    :return: lista filtrata
    """
    filtered_list = []
    for expense in list:
        if expenses.get_expense_value(expense) >= value:
            filtered_list.append(expense)

    return filtered_list
