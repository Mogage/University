from domain.data import Books, Rent


class BookFileRepository:
    """
        Clasa pentru stocarea entitatilor de tip carte
    """
    def __init__(self, file_name):
        """
            Initializarea unui repository pentru carti
        :param file_name: string - contine path-ul catre fisierul din care luam date
        """
        self.__books = []
        self.__file_name = file_name
        self.__load_from_file()

    @property
    def books(self):
        """
            Getter pentru cartile din repository
        :return: list[] - lista cu carti
        """
        return self.__books

    @staticmethod
    def __create_data_from_line(line):
        """
            Preia o linie din fisier si extrage datele din ea
        :param line: string - linie din fisier
        :return: Books - entitate de tip carte
        """
        fields = line.split(',')
        id = int(fields[0])
        title = fields[1]
        author = fields[2]
        year = int(fields[3])
        return Books(id, title, author, year)

    def __load_from_file(self):
        """
            Citeste liniile din fisier si adauga in memorie cartile corespunzatoare
        """
        with open(self.__file_name, "r") as books_input:
            for line in books_input:
                book = self.__create_data_from_line(line)
                self.__books.append(book)


class RentFileRepository:
    """
        Clasa pentru stocarea inchirierilor
    """
    def __init__(self, file_name):
        """
            Initializarea unui repository pentru inchirieri
        :param file_name: string - contine path-ul catre fisierul din care luam date
        """
        self.__rents = []
        self.__file_name = file_name
        self.__load_from_file()

    @property
    def rents(self):
        """
            Getter pentru inchirierile din repository
        :return:
        """
        return self.__rents

    @staticmethod
    def __create_data_from_line(line):
        """
            Preia o linie din fisier si extrage datele din ea
        :param line: string - linie din fisier
        :return: Rent - entitate de tip rent
        """
        fields = line.split(',')
        id = int(fields[0])
        id_book = int(fields[1])
        date = fields[2]
        duration = int(fields[3])
        return Rent(id, id_book, date, duration)

    def __load_from_file(self):
        """
            Citeste liniile din fisier si adauga in memorie inchirierile corespunzatoare
        """
        with open(self.__file_name, "r") as rent_input:
            for line in rent_input:
                rent = self.__create_data_from_line(line)
                self.__rents.append(rent)
