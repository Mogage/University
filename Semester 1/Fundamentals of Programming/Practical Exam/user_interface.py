from errors import EventError


class Ui:
    """
        Clasa pentru interfata utilizator
    """
    def __init__(self, service):
        """
            Functia de initializare, se afiseaza evenimentele din ziua actuala
        :param service: un service de evenimente
        """
        self.__service = service
        self.__print_events_from_today()

    def __print_events_from_today(self):
        today_events = self.__service.get_today_events()

        for _event in today_events:
            print(_event)

    @staticmethod
    def __show_menu():
        """
            Se afiseaza meniul din care alege utilizatorul
        """
        print(" Meniu:\n"
              "\t Introduceti 1 pentru a adauga un eveniment.\n"
              "\t Introduceti 2 pentru a afisa evenimentele de la o anumita data.\n"
              "\t Introduceti 3 pentru a exporta evenimentele intr-un fisier.\n"
              "\t Introduceti 4 pentru a afisa meniul.\n"
              "\t Introduceti 0 pentru a iesi din aplicatie.")

    def __add_event(self):
        """
            Si citesc datele unui eveniment si se adauga respectiul eveniment
        """
        event_date = input("Introduceti data evenimentului in formatul dd.mm.yyyy:  ")
        event_hour = input("Introduceti ora evenimentului in formatul hh:mm:  ")
        event_description = input("Introduceti descrierea evenimentului: ")

        self.__service.add_event(event_date, event_hour, event_description)

    def __show_events_from_date(self):
        """
            Afiseaza evenimentele de la o anumita data citita
        """
        event_date = input("Introduceti data evenimentelor in formatul dd.mm.yyyy:  ")

        events_from_date = self.__service.get_events_from_date(event_date)

        for _event in events_from_date:
            print(_event)

    def __export_to_file(self):
        """
            Exporta intr-un fisier fara extensie citit de la utiizator evenimentele a caror descriere contin un sir citit de la utilizator
        """
        file_path = input("Introduceti numele fisierului in care sa se exporte evenimentele fara extensie: ")
        description = input("Introduceti descrierea dupa care vor fi exportate evenimentele: ")

        self.__service.export_to_file(file_path, description)

    def run_main(self):
        """
            Gestioneaza interfata utiizator
        """
        self.__show_menu()
        while True:
            user_input = input(">>>")

            if user_input == "":
                continue
            if user_input == "0":
                return

            if user_input == "1":
                try:
                    self.__add_event()
                except EventError as ev_e:
                    print(str(ev_e))
            elif user_input == "2":
                try:
                    self.__show_events_from_date()
                except EventError as ev_e:
                    print(str(ev_e))
            elif user_input == "3":
                try:
                    self.__export_to_file()
                except IOError:
                    print("Eroare la exportarea evenimentelor")
            elif user_input == "4":
                self.__show_menu()
            else:
                print("Valoare introdusa invalida.")
