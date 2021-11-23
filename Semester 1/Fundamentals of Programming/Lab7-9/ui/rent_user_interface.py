"""
    Author: Mogage Nicolae
    Creation date: 16 nov 2021
    Modul pentru partea de interactiune cu utilizatorul in meniul de inchiriere
"""
from error.errors import RentRepositoryError, RentError

class RentUI:
    """
        Clasa pentru interactiunea cu utilizatorul in cadrul meniului pentru inchirieri
    """
    def __init__(self, rent_service):
        """
            Initializarea datelor pentru partea de interactiune cu utilizatorul
        :param rent_service: serviciu inchirieri
        """
        self.__rent_service = rent_service

    @staticmethod
    def __rent_menu():
        """
            Afisarea meniulului principal de inchirieri
        """
        print(" Rent menu:\n"
              "\t -rent book pentru a inchiria o carte.\n"
              "\t -generate pentru a efectua un numar de inchirieri aleatorii.\n"
              "\t -return pentru a inapoia o carte.\n"
              "\t -print pentru a afisa inchirierile.\n"
              "\t -show menu pentru a afisa meniul.\n"
              "\t -main menu pentru a va intoarce la meniul principal.\n"
              "\t -exit pentru a iesi din program.")

    @staticmethod
    def __print_menu():
        """
            Afisarea meniului de tiparire inchirieri
        """
        print(" Print rent menu:\n"
              "\t -all pentru a afisa toate inchirierile.\n"
              "\t -client pentru a afisa toate inchirerile facute de un anumit client.\n"
              "\t -rent menu pentru a va intoarce la meniul de inchirieri.\n"
              "\t -main menu pentru a va intoarce la meniul principal.\n"
              "\t -exit pentru a iesi din program.")

    def __ui_add_rent(self):
        """
            Adaugarea unei inchirieri facute de un client existent cu un id - int care inchiriaza
            o carte existenta cu un id - int
        """
        try:
            id_client = int(input("Id-ul clientului care inchiriaza cartea: "))
        except ValueError:
            print("Valoare numerica invalida.")
            return

        self.__rent_service.check_client_exists(id_client)

        try:
            id_book = int(input("Id-ul cartii care se inchiriaza: "))
        except ValueError:
            print("Valoare numerica invalida.")
            return

        self.__rent_service.check_book_exists(id_book)

        self.__rent_service.add_rent(id_book, id_client)
        print("Carte inchiriata cu succes.")

    def __ui_delete_rent(self):
        """
            Stergerea unei inchirieri facute de un client existent cu un id - int care a inchiriat
            o carte existenta cu un id - int
        :raise RentRepositoryError: Daca nu s-a efectuat nicio inchiriere
        """
        rent = self.__rent_service.get_all_rent()
        if len(rent) == 0:
            raise RentRepositoryError("Nu s-a inchiriat nicio carte.")

        try:
            id_client = int(input("Id-ul clientului care returneaza cartea: "))
        except ValueError:
            print("Valoare numerica invalida.")
            return

        self.__rent_service.check_rent_client_exists(id_client)

        try:
            id_book = int(input("Id-ul cartii care se returneaza: "))
        except ValueError:
            print("Valoare numerica invalida.")
            return

        self.__rent_service.delete_rent(id_book, id_client)
        print("Carte returnata cu succes.")

    def __ui_print(self):
        """
            Gestiune meniu pentru afisarea inchirieri
        :raise RentRepositoryError: Daca nu s-a efectuat nicio inchiriere
        """
        rent = self.__rent_service.get_all_rent()
        if len(rent) == 0:
            raise RentRepositoryError("Nu s-a inchiriat nicio carte.")

        self.__print_menu()
        while True:
            user_input = input(">>>").strip()

            if user_input == "exit":
                return "exit"
            if user_input == "main menu":
                return "main"
            if user_input == "rent menu":
                return
            if user_input == "all":
                self.__rent_service.print_all()
            elif user_input == "client":
                try:
                    id_client = int(input("Id client: "))
                    self.__rent_service.print_rent_by_client_id(id_client)
                except ValueError:
                    print("Valoare numerica invalida.")
                except RentError as re:
                    print(re)
            else:
                print("Comanda invalida.")

    def __ui_generate_rent(self):
        try:
            number_of_rent = int(input("Numarul de inchirieri care sa se genereze: "))
        except ValueError:
            print("Valoare numerica invalida.")
            return

        self.__rent_service.generate_rent(number_of_rent)

    def rent_run(self):
        """
            Meniul principal pentru gestiune inchirieri
        """
        self.__rent_menu()
        while True:
            user_input = input(">>>").strip()

            if user_input == "exit":
                return "exit"
            if user_input == "main menu":
                return
            if user_input == "":
                continue
            if user_input == "show menu":
                self.__rent_menu()
            elif user_input == "rent book":
                try:
                    self.__ui_add_rent()
                except RentError as re:
                    print(re)
            elif user_input == "generate":
                try:
                    self.__ui_generate_rent()
                except RentError as re:
                    print(re)
            elif user_input == "return":
                try:
                    self.__ui_delete_rent()
                except RentError as re:
                    print(re)
            elif user_input == "print":
                output = self.__ui_print()
                if output == "exit":
                    return True
                if output == "main":
                    return
                self.__rent_menu()
            else:
                print("Comanda invalida")
