class Ui:
    """
        Clasa pentru interactiunea cu utilizatorul
    """
    def __init__(self, books_service, rent_service):
        """
            Initializarea atributelor unui ui
        :param books_service: BooksService
        :param rent_service: RentService
        """
        self.__books_service = books_service
        self.__rent_service = rent_service

    @staticmethod
    def __menu():
        """
            Afiseaza meniul cu care interactioneaza utilizatorul
        """
        print(" Menu:\n"
              "\t -1 pentru a afisa cerinta 1.\n"
              "\t -2 pentru a afisa cerinta 2.\n"
              "\t -show menu pentru a afisa meniul.\n"
              "\t -exit pentru a iesi din program.")

    def __requirement_1(self):
        """
            Citeste un string de la tastatura si afiseaza cartile a caror titlu se termina cu acel string
        """
        while True:
            user_input = input("Introduceti sirul de caractere cu care sa se termine titlul cartii: ")

            # Daca sirul e gol mai cerem inca o data sa se introduca alt sir
            if user_input == "":
                continue

            filtered_books = self.__books_service.requirement_1(user_input)

            if len(filtered_books) == 0:
                print("Nu exista nicio carte care sa se termine cu acest sir.")
                return

            for _book in filtered_books:
                print(str(_book))

            return

    def __requirement_2(self):
        """
            Citeste un int > 0 de la tastatura si afiseaza inchirierile a caror durata imprumut in zile este egal cu
            acel int
        """
        try:
            user_input = int(input("Introduceti numarul de zile: "))
        except ValueError:
            print("Valoare introdusa trebuie sa fie un numar")
            return

        if user_input < 0:
            print("Numarul trebuie sa fie pozitiv")
            return

        filtered_rents = self.__rent_service.requirement_2(user_input)

        if len(filtered_rents) == 0:
            print("Nu exista nicio inchiriere cu acest numar de zile.")
            return

        for _rent in filtered_rents:
            print(str(_rent))

    def run(self):
        """
            Gestioneaza interactioneaza cu utilizatorul
        """
        self.__menu()
        while True:
            user_input = input(">>>")

            if user_input == "exit":
                return
            if user_input == "":
                continue

            if user_input == "1":
                self.__requirement_1()
            elif user_input == "2":
                self.__requirement_2()
            elif user_input == "show menu":
                self.__menu()
            else:
                print("Comanda invalida.")
