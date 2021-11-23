"""
    Author: Mogage Nicolae
    Creation date: 22 nov 2021
    Modul pentru partea de interactiune cu utilizatorul pentru rapoarte
"""
from error.errors import RentRepositoryError

class QueryUI:
    """
        Clasa pentru interactiunea cu utilizatorul in meniul de rapoarte
    """
    def __init__(self, rent_service):
        """
            Initializarea datelor pentru partea de interactiune cu utilizatorul
        :param rent_service: serviciu inchirieri
        """
        self.__rent_service = rent_service

    @staticmethod
    def __query_menu():
        """
            Afisarea meniului de rapoarte
        """
        print(" Query menu:\n"
              "\t -book pentru a afisa cele mai inchiriate carti.\n"
              "\t -client pentru a afisa clientii cu cele mai multe carti inchiriate.\n"
              "\t -top pentru a afisa primii 20% cei mai activi clienti.\n"
              "\t -show menu pentru a afisa meniul.\n"
              "\t -main menu pentru a va intoarce la meniul principal.\n"
              "\t -exit pentru a iesi din program.")

    def query_run(self):
        """
            Meniun principal pentru lucrul cu rapoarte
        """
        number_of_rent = len(self.__rent_service.get_all_rent())
        if number_of_rent == 0:
            raise RentRepositoryError("Nu s-a inchiriat nicio carte.")
        self.__query_menu()

        while True:
            user_input = input(">>>").strip()

            if user_input == "exit":
                return True
            if user_input == "main menu":
                return
            if user_input == "":
                continue
            if user_input == "book":
                self.__rent_service.print_most_rented_books()
            elif user_input == "client":
                self.__rent_service.print_most_active_clients()
            elif user_input == "top":
                self.__rent_service.print_most_active_clients(top=True)
            elif user_input == "show menu":
                self.__query_menu()
            else:
                print("Comanda invalida.")
