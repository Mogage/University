import expenses
import service

class UiMenu:
    """

    """
    def __init__(self):
        """

        """
        self.__list = []
        self.__commands_list = ["add", "delete", "print", "filter", "menu", "exit"]

    @staticmethod
    def __print_menu():
        """
            Afiseaza meniul general cu care interactioneaza utilizatorul
        :return: -
        """
        print("    Meniu:\n"
              "\t   add [zi] [tip] [suma] pentru a adauga o noua cheltuiala.\n"
              "\t   print all pentru a tipari toate cheltuielile.\n"
              "\t   print [tip] pentru a tipari cheltuielile de un anumit tip.\n"
              "\t   delete [tip] pentru a sterge cheltuielile de un anumit tip.\n"
              "\t   menu pentru a afisa meniul.\n"
              "\t   exit pentru a iesi din program.")

    @staticmethod
    def __validate(day, value):
        """
            Verifica daca day si value sunt de tip int, iar daca nu ridica exceptii
            day - "Ziua introdusa este invalida."
            value - "Suma introdusa este invalida."
        :param day: un string de la tastatura
        :param value: un string de la tastatura
        :return: int(day), int(value)
        """
        try:
            int(day)
        except ValueError:
            raise Exception("Ziua introdusa este invalida.")
        try:
            int(value)
        except ValueError:
            raise Exception("Suma introdusa este invalida.")
        return int(day), int(value)

    @staticmethod
    def __get_data(command):
        """
            Verifica daca datele de intrare exista si sunt nevide, iar daca da le returneaza separat
        :return: string, string, string
        :raise: Exception("Comanda invalida")
        """
        command = command.strip()
        auxiliary_list = command.split(" ")

        if len(auxiliary_list) == 3:
            return auxiliary_list[0], auxiliary_list[1], auxiliary_list[2]
        else:
            raise Exception("Comanda invalida")

        if day == '' or type == '' or value == '':
            raise Exception("Comanda invalida")

    def __add_expense(self, command):
        """
            Se obtin datele cheltuielii si se adauga in lista
        :param command: o comanda
        :return: -
        """
        command = command[4:]

        try:
            day, type, value = self.__get_data(command)
        except Exception as ex:
            print(ex)
            return

        try:
            day, value = self.__validate(day, value)
        except Exception as ex:
            print(ex)
            return

        try:
            service.srv_add_expense_to_list(self.__list, day, value, type)
            print("Cheltuiala inregistrata cu succes")
        except Exception as ex:
            print(ex)

    @staticmethod
    def __command_prefix(command):
        """
            Returneaza primul cuvant dintr-o comanda, cuvintele fiind separate cu spatii
        :param command: o comanda de la utilizator
        :return: prefix
        """
        space = command.find(' ')
        if space != -1:
            return command[:space]
        return command

    def __print_expenses(self):
        """
             Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
             Altfel tipareste elementele din lista in formatul prestabilit
        :return: -
        """
        if service.srv_empty_list(self.__list):
            print("Nu exista nicio cheltuiala inregistrata.")
            return

        print(service.srv_print_expenses(self.__list))

    def __print_type(self, command):
        """
            Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
            Altfel se citeste tipul cheltuielilor care sa se tipareasca si le tipareste daca tipul este valid
        :param command: o comanda
        :return: -
        """
        if service.srv_empty_list(self.__list):
            print("Nu exista nicio cheltuiala inregistrata.")
            return

        type = command[6:]

        if expenses.check_type(type):
            print(service.srv_print_expense_by_type(self.__list, type))
            return
        else:
            print("Tip introdus invalid.")

    def __delete_type(self, command):
        """
             Afiseaza mesajul "Nu exista nicio cheltuiala inregistrata." daca lista este goala
             Altfel se citeste tipul cheltuielilor care sa se stearga si le sterge daca tipul este valid
        :param command: o comanda
        :return: -
        """
        if service.srv_empty_list(self.__list):
            print("Nu exista nicio cheltuiala inregistrata.")
            return

        type = command[7:]

        if expenses.check_type(type):
            print(service.srv_delete_expense_by_type(self.__list, type))
            return
        else:
            print("Tip introdus invalid.")

    def __ui_input(self, command):
        """
            Functie prin care se prelucreaza sirul de comenzi
            - add [zi] [tip] [suma] pentru a adauga o noua cheltuiala.
            - print all pentru a tipari toate cheltuielile.
            - print [tip] pentru a tipari cheltuielile de un anumit tip.
            - delete [tip] pentru a sterge cheltuielile de un anumit tip.
            - menu pentru a afisa meniul.
            - exit pentru a iesi din program.
        :return: -
        """
        list_with_separated_commands = command.split(", ")
        for command_index in list_with_separated_commands:
            auxiliary_command = self.__command_prefix(command_index)
            if auxiliary_command == "exit":
                return "exit"
            if auxiliary_command == "menu":
                self.__print_menu()
            elif auxiliary_command == "add":
                self.__add_expense(command_index)
            elif command_index == "print all":
                self.__print_expenses()
            elif auxiliary_command == "print":
                self.__print_type(command_index)
            elif auxiliary_command == "delete":
                self.__delete_type(command_index)
            else:
                print("Comanda invalida.")

    def ui_user(self):
        """
            Functie prin care se interactioneaza cu utilizatorul, aici se afiseaza meniul si se introduce optiunea utilizatorului
        :return: -
        """
        self.__print_menu()
        while True:
            comand = input(">>>")
            if self.__ui_input(comand) == "exit":
                return
