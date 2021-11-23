"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru partea de interactiune cu utilizatorul
"""
from error.errors import RentRepositoryError
from ui.book_user_interface import BookUI
from ui.client_user_interface import ClientUI
from ui.rent_user_interface import RentUI
from ui.query_user_interface import QueryUI

class Console:
    """
        Clasa pentru interactiunea cu utilizatorul in cadrul meniului principal
    """
    def __init__(self, books_service, clients_service, rent_service):
        """
            Initializeaza meniurile cu care interactioneaza utilizatorul din cel principal
        :param books_service: serviciu de carti
        :param clients_service: serviciu de clienti
        """
        self.__rent_ui = RentUI(rent_service)
        self.__books_ui = BookUI(books_service)
        self.__clients_ui = ClientUI(clients_service)
        self.__query_ui = QueryUI(rent_service)

    @staticmethod
    def __main_menu():
        """
            Afisarea meniului principal
        """
        print(" Main menu:\n"
              "\t -book pentru a accesa meniul de gestiune carti.\n"
              "\t -client pentru a accesa meniul de gestiune clienti.\n"
              "\t -rent pentru a accesa meniul de inchiriere carti.\n"
              "\t -query pentru a afisa meniul de rapoarte.\n"
              "\t -exit pentru a iesi din program")

    def run(self):
        """
            Gestiune meniul principal
        """
        self.__main_menu()
        while True:
            user_input = input(">>>")
            user_input = user_input.strip()

            if user_input == "exit":
                return
            if user_input == "":
                continue
            if user_input == "book":
                exit = self.__books_ui.book_run()
                if exit:
                    return
                self.__main_menu()
            elif user_input == "client":
                exit = self.__clients_ui.client_run()
                if exit:
                    return
                self.__main_menu()
            elif user_input == "rent":
                exit = self.__rent_ui.rent_run()
                if exit:
                    return
                self.__main_menu()
            elif user_input == "query":
                try:
                    exit = self.__query_ui.query_run()
                except RentRepositoryError as rre:
                    print(rre)
                    continue
                if exit:
                    return
                self.__main_menu()
            else:
                print("Comanda invalida")
