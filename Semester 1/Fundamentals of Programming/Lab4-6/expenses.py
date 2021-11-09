def create_expense(day, value, type):
    """
        Creaza o cheltuiala care retine ziua day, suma value si tipul type
    :param day: int >= 1 and <= 31, reprezinta ziua unei zile
    :param value: int >= 1 and <= 9999, reprezinta suma cu care s-a efectuat o cheltuiala
    :param type: string, reprezinta tipul cheltuielii si trebuie sa fie una din ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    :return: O cheltuiala cu ziua day, suma value si tipul type
    """
    """return {
        "day":   day,
        "type": type,
        "value": value
    }"""
    return [day, type, value]

def get_expense_day(expense):
    """
        Returneaza ziua cheltuielii expense
    :param expense: o cheltuiala
    :return: int >= 1 and <= 31, ziua cheltuielii expense
    """
    # return expense["day"]
    return expense[0]

def get_expense_value(expense):
    """
        Returneaza suma cheltuielii expense
    :param expense: o cheltuiala
    :return: int >= 1 and <= 9999, suma cheltuielii expense
    """
    # return expense["value"]
    return expense[2]

def get_expense_type(expense):
    """
        Returneaza tipul cheltuielii expense
    :param expense: o cheltuiala
    :return: string din lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"], tipul cheltuielii expense
    """
    # return expense["type"]
    return expense[1]

def to_str_expense(expense):
    """
        Transforma o cheltuiala in string pentru afisare
    :param expense: o cheltuiala
    :return: string de tipul: Ziua: day Tip: type Suma: value.
    """
    return "Ziua: " + str(get_expense_day(expense)) + " Tip: " + str(get_expense_type(expense)) + " Suma: " + str(get_expense_value(expense)) + "."

def check_type(type):
    """
        Verifica daca tipul unei cheltuieli este valid sau nu
    :param type: Tipul unei cheltuieli
    :return: True daca tipul se afla in lista, False altfel
    """
    type = type.lower()

    for index_type in ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]:
        if type == index_type:
            return True
    return False

def check_expense(expense):
    """
        Verifica daca o cheltuiala este valida sa unu
    :param expense: o cheltuiala
    :return: -
    :raises: Exception cu textul:
            "Zi invalida.\n", daca day < 1 or day > 31
            "Suma invalida.\n", daca value < 1 or value > 9999
            "Tip invalid.\n", daca tipul nu se incadreaza in lista ["mancare", "intretinere", "imbracaminte", "telefon", "altele"]
    """
    error = ""
    day = get_expense_day(expense)
    value = get_expense_value(expense)
    if day < 1 or day > 31:
        error += "Ziua introdusa este invalida.\n"
    if not check_type(get_expense_type(expense)):
        error += "Tipul introdus este invalid.\n"
    if value < 1 or value > 9999:
        error += "Suma introdusa este invalida.\n"
    if len(error) > 0:
        raise Exception(error)

def equal_expenses(_expense, expense):
    """
        Verifica daca 2 cheltuieli sunt la fel
    :param _expense: o cheltuiala
    :param expense: o cheltuiala
    :return: True daca cele 2 cheltuieli au aceeasi zi si acelasi tip, False altfel
    """
    return True if (get_expense_day(_expense) == get_expense_day(expense) and get_expense_type(_expense) == get_expense_type(expense)) else False

def add_expense_to_list(list, expense):
    """
        Adauga o cheltuiala la lista sortata constant dupa zi
    :param list: lista de cheltuieli
    :param expense: o cheltuiala
    :return: -
    :raise Exception cu textul:
            "Cheltuiala existent.\n", daca exista o cheltuiala de acelasi tip in aceeasi zi
    """
    for _expense in list:
        if equal_expenses(_expense, expense):
            raise Exception("Cheltuiala existenta.\n")

    list_index = 0
    list_length = len(list)
    day = get_expense_day(expense)
    while list_index < list_length and get_expense_day(list[list_index]) <= day:
        list_index += 1

    list.insert(list_index, expense)
