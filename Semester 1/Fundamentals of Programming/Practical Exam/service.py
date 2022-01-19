import datetime

from data import Events


class Service:
    """
        Clasa de gestiune a evenimentelor
    """
    def __init__(self, validator, repository):
        """
            Initializeaza un service cu un Repository si un Validator
        :param validator: Validator()
        :param repository: Repository()
        """
        self.__validator = validator
        self.__repository = repository

    def add_event(self, date, hour, description):
        """
            Construieste un eveniment, iar daca este corect il adauga in memorie
        :param date: string
        :param hour: string
        :param description: string
        """
        event = Events(date, hour, description)
        self.__validator.validate_event(event)
        self.__repository.add_event(event)

    def get_today_events(self):
        """
            Construieste o lista cu toate evenimentele din ziua in care se ruleaza aplicatia si o sorteaza dupa ora evenimentelor
        :return: list, evenimentele din ziua actuala sortate dupa ora
        """
        today_day = str(datetime.date.today().day)
        today_month = str(datetime.date.today().month)
        today_year = str(datetime.date.today().year)
        if len(today_day) == 1:
            today_day = '0' + today_day
        if len(today_month) == 1:
            today_month = '0' + today_month

        today_date = today_day + '.' + today_month + '.' + today_year
        today_events = []
        for _event in self.__repository.all_events:
            if _event.data == today_date:
                today_events.append(_event)

        return sorted(today_events, key=lambda event: event.ora)

    def get_events_from_date(self, date):
        """
            Construieste o lista cu toate evenimentele de la o anumita data
        :param date: string
        :return: list, evenimentele de la data respectiva
        """
        self.__validator.validate_date(date)
        events_from_date = []
        for _event in self.__repository.all_events:
            if _event.data == date:
                events_from_date.append(_event)

        return events_from_date

    def get_all_events(self):
        """
        :return: list, toate evenimentele
        """
        return self.__repository.all_events

    @staticmethod
    def __create_line_from_event(event):
        """
            Construieste dintr-o entitate de tip Events un string pentru a fi scris in fisier
        :param event: entitate de tip Events
        :return: string, in formatul "{data};{ora};{descriere}"
                """
        line = ""
        line += event.data + ';'
        line += event.ora + ';'
        line += event.descriere + '\n'
        return line

    def __get_events_with_description(self, description):
        """
            Cosntruieste o lista cu evenimentele a caror descriere contin sirul description
        :param description: string
        :return: list, evenimentele a caror descriere contin sirul description
        """
        events = []

        for _event in self.__repository.all_events:
            if description in _event.descriere:
                events.append(_event)

        return events

    @staticmethod
    def __add_extension(file_path):
        """
            Modifica extensia fisierului file_path in txt, iar daca nu exista o adauga
        :param file_path: string, calea catre un fisier
        :return: string, calea catre un fisier cu extensia .txt
        """
        if '.' in file_path:
            file_path = file_path[:file_path.find('.')]

        file_path += ".txt"

        return file_path

    def export_to_file(self, file_path, description):
        """
            Scrie in fisierul file_path evenimentele sortate dupa data si ora a caror descriere contin sirul description
        :param file_path: string, calea catre un fisier
        :param description: string
        """

        file_path = self.__add_extension(file_path)

        events_to_export = self.__get_events_with_description(description)

        sorted_events_to_export = sorted(events_to_export, key=lambda event: (event.data, event.ora))

        with open(file_path, "w") as file:
            for _event in sorted_events_to_export:
                line_to_export = self.__create_line_from_event(_event)
                file.write(line_to_export)
