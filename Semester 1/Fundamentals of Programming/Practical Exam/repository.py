from errors import RepositoryError
from data import Events


class _InMemoryRepository:
    """
        Clasa pentru gestionarea evenimentelor in timp ce ruleaza aplicatia
    """
    def __init__(self):
        self.__events_list = []

    @property
    def all_events(self):
        """
            Getter pentru lista cu toate evenimentele
        :return: lista, lista cu evenimentele
        """
        return self.__events_list[:]

    def add_event(self, event):
        """
            Adauga un event in memorie
        :param event: entitate de tip Events
        """
        self.__events_list.append(event)


class FileRepository(_InMemoryRepository):
    """
        Clasa pentru gestiuena evenimentelor cu fisiere
    """
    def __init__(self, file_path):
        _InMemoryRepository.__init__(self)
        self.__file_path = file_path
        self.__load_from_file()

    @staticmethod
    def __create_line_from_data(event):
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

    @staticmethod
    def __create_data_from_line(line):
        """
            Construiesti dintr-o linie din fisier o entitate de tip Events
        :param line: string, linie din fisier
        :return: entitate de tip Events
        """
        line = line.removesuffix("\n")
        fields = line.split(';')
        if len(fields) == 3:
            event_date = fields[0]
            event_hour = fields[1]
            event_description = fields[2]
            event = Events(event_date, event_hour, event_description)
            return event

        raise RepositoryError

    def __load_from_file(self):
        """
            Incarca in memorie toate evenimentele existente din fisierul in care sunt stocate
        """
        with open(self.__file_path, "r") as file:
            for line in file:
                event = self.__create_data_from_line(line)
                _InMemoryRepository.add_event(self, event)

    def __append_file(self, event):
        """
            Adauga la fisier o noua entiate de tip Events
        :param event: entitate de tip Events
        """
        with open(self.__file_path, "a") as file:
            line = self.__create_line_from_data(event)
            file.write(line)

    def add_event(self, event):
        """
            Adauga un eveniment in memorie si in fisier
        :param event: entitate de tip Events
        """
        _InMemoryRepository.add_event(self, event)
        self.__append_file(event)
