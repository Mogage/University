class Punct:
    def __init__(self, coordonata_x, coordonata_y):
        self.__coordonata_x = coordonata_x
        self.__coordonata_y = coordonata_y

    @property
    def x(self):
        return self.__coordonata_x

    @property
    def y(self):
        return self.__coordonata_y

    def __gt__(self, other):
        if self.__coordonata_x != other.x:
            return self.__coordonata_x > other.x

        return self.__coordonata_y > other.y

    def __eq__(self, other):
        return self.__coordonata_x == other.x and self.__coordonata_y == other.y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"


def determinant(punct_1, punct_2, punct_3):
    value = punct_1.x * punct_2.y + punct_2.x * punct_3.y + punct_3.x * punct_1.y
    value = value - (punct_3.x * punct_2.y + punct_2.x * punct_1.y + punct_1.x * punct_3.y)
    return value


def format_afisare_solutie(numar_puncte_stiva, stiva):
    global lista_puncte
    string_to_return = '{'
    _punct = 1
    while _punct < numar_puncte_stiva:
        string_to_return = string_to_return + str(lista_puncte[stiva[_punct - 1]])
        string_to_return = string_to_return + "; "
        _punct = _punct + 1
    string_to_return = string_to_return + str(lista_puncte[stiva[_punct - 1]])
    string_to_return = string_to_return + '}'
    return string_to_return


def consistent(numar_puncte_stiva, punct, stiva):
    global used, lista_puncte
    if 0 == numar_puncte_stiva:
        return True
    if not used[punct] and lista_puncte[punct] > lista_puncte[stiva[-1]]:
        return True
    return False


def consistent_iterativ(numar_puncte_stiva, punct, stiva):
    global used, lista_puncte
    if 1 == numar_puncte_stiva:
        return True
    if lista_puncte[punct] > lista_puncte[stiva[-2]]:
        return True
    return False


def solutie(numar_puncte_stiva, stiva):
    global lista_puncte
    if numar_puncte_stiva < 3:
        return False

    for _punct_1 in range(0, numar_puncte_stiva):
        for _punct_2 in range(_punct_1 + 1, numar_puncte_stiva):
            for _punct_3 in range(_punct_2 + 1, numar_puncte_stiva):
                valoare_determinat = determinant(lista_puncte[stiva[_punct_1]], lista_puncte[stiva[_punct_2]],
                                                 lista_puncte[stiva[_punct_3]])
                if valoare_determinat == 0:
                    return True

    return False


def backtracking(back_stack):
    global numar_puncte, lista_puncte, used, exista_solutie
    for _punct in range(numar_puncte):  # Candidat e orice element intrucat trebuie generate toate submultimile
        numar_puncte_in_stiva = len(back_stack)
        if consistent(numar_puncte_in_stiva, _punct, back_stack):  # Verificam candidatul sa fie consistent
            used[_punct] = True
            back_stack.append(_punct)   # Extindem solutia candidat daca e consistent
            numar_puncte_in_stiva = numar_puncte_in_stiva + 1
            if solutie(numar_puncte_in_stiva, back_stack):  # Daca s-a gasit solutia se afiseaza
                print(format_afisare_solutie(numar_puncte_in_stiva, back_stack))
                exista_solutie = True
            if numar_puncte_in_stiva < numar_puncte:
                backtracking(back_stack)
            back_stack.pop()
            used[_punct] = False


def backtracking_iterativ():
    global numar_puncte, lista_puncte, exista_solutie
    stack = [-1]  # Solutia candidat
    while len(stack) > 0:
        choosed = False
        numar_puncte_in_stiva = len(stack)
        while stack[-1] < numar_puncte - 1 and not choosed:
            stack[-1] = stack[-1] + 1
            choosed = consistent_iterativ(numar_puncte_in_stiva, stack[-1], stack)  # Verificam candidatul sa fie
            # consistent
        if choosed:
            if solutie(numar_puncte_in_stiva, stack):  # Daca s-a gasit solutia se afiseaza
                print(format_afisare_solutie(numar_puncte_in_stiva, stack))
                exista_solutie = True
            stack.append(-1)  # Extindem solutia candidat
        else:
            stack.pop(numar_puncte_in_stiva - 1)  # Scoatem ultimul element de pe stiva


def keyboard_input():
    global numar_puncte, lista_puncte
    numar_puncte = 0
    while True:
        try:
            numar_puncte = int(input("Introduceti numarul de puncte in plan: "))
        except ValueError:
            print("Trebuie introdus un numar intreg.")
            continue

        if numar_puncte < 3:
            print("Trebuiesc cel putin 3 puncte.")
            continue

        break

    lista_puncte = []
    for _numar_punct in range(numar_puncte):
        print("Punctul " + str(_numar_punct + 1) + ":")
        x_axis = y_axis = '\0'
        while True:
            try:
                x_axis = float(input("Introduceti coordonata x: "))
            except ValueError:
                print("Trebuie introdus un numar real.")
                continue

            try:
                y_axis = float(input("Introduceti coordonata y: "))
            except ValueError:
                print("Trebuie introdus un numar real.")
                continue

            break

        punct = Punct(x_axis, y_axis)
        lista_puncte.append(punct)


def file_input():
    global numar_puncte, lista_puncte
    lista_puncte = []
    with open("input.txt", "r") as input:
        numar_puncte = int(input.readline())
        for line in input:
            line = line.removesuffix('\n')
            data = line.split(' ')
            punct = Punct(float(data[0]), float(data[1]))
            lista_puncte.append(punct)


def read_input():
    while True:
        user_input = input("De unde se citesc datele(file / keyboard): ")
        if user_input == "file":
            file_input()
            return
        elif user_input == "keyboard":
            keyboard_input()
            return
        else:
            print("Comanda invalida.")


if __name__ == '__main__':
    global numar_puncte, lista_puncte
    exista_solutie = False
    read_input()
    used = [False for x in range(numar_puncte)]
    print("-------------------------------RECURSIV----------------------------------------------")
    backtracking([])
    if not exista_solutie:
        print("Nu exista puncte coliniare.")
    print("-------------------------------ITERATIV----------------------------------------------")
    exista_solutie = False
    backtracking_iterativ()
    if not exista_solutie:
        print("Nu exista puncte coliniare.")
