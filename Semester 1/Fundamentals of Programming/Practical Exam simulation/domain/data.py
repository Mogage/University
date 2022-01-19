class Books:
    """
        Clasa pentru entitatile de tip carte
    """

    def __init__(self, id, title, author, year):
        """
            Initializeaza o carte cu atributele id, nume, autor si an
        :param id: int
        :param title: string
        :param author: string
        :param year: int
        """
        self.__id = id
        self.__title = title
        self.__author = author
        self.__year = year

    @property
    def id(self):
        """
            Getter pentru id-ul unei carti
        :return: int > 0, id-ul unei carti
        """
        return self.__id

    @property
    def title(self):
        """
            Getter pentru titlul unei carti
        :return: string, titlul unei carti
        """
        return self.__title

    @title.setter
    def title(self, other_title):
        """
            Setter pentru titlul unei carti
        :param other_title: string - alt titlu
        """
        self.__title = other_title

    @property
    def author(self):
        """
            Getter pentru autorul unei carti
        :return: string, autorul unei carti
        """
        return self.__author

    @author.setter
    def author(self, other_author):
        """
            Setter pentru autorul unei carti
        :param other_author: string - alt autor
        """
        self.__author = other_author

    @property
    def year(self):
        """
            Getter pentru anul unei carti
        :return: int > 0, anul unei carti
        """
        return self.__year

    @year.setter
    def year(self, other_year):
        """
            Setter pentru anul unei carti
        :param other_year: string - alt an
        """
        self.__year = other_year

    def __eq__(self, other):
        """
            2 carti sunt egale cand id-ul este egal
        :param other: int
        :return: True daca sunt egale, false altfel
        """
        return self.__id == other.id

    def __str__(self):
        """
            Rescrie formatul pentru string al unei carti
        :return: string - sirul formatat pentru afisarea unei carti
        """
        return str(self.__id) + ": Cartea '" + self.__title + "' scrisa de '" + self.__author + "' in anul " + str(self.__year) + "."


class Rent:
    """
        Clasa pentru entitati inchirieri
    """

    def __init__(self, id, id_book, date, duration):
        """
            Initializeaza o inchiriere cu atributele id, id_book, date, duration
        :param id: int
        :param id_book: int
        :param date: string
        :param duration: int
        """
        self.__id = id
        self.__id_book = id_book
        self.__date = date
        self.__duration = duration

    @property
    def id(self):
        """
            Getter pentru id-ul unei inchirieri
        :return: int
        """
        return self.__id

    @property
    def id_book(self):
        """
            Getter pentru id-ul unei carti
        :return: int
        """
        return self.__id_book

    @property
    def date(self):
        """
            Getter pentru data unei inchirieri
        :return: string
        """
        return self.__date

    @date.setter
    def date(self, other_date):
        """
            Setter pentru data unei inchirieri
        :param other_date: string - alta data
        """
        self.__date = other_date

    @property
    def duration(self):
        """
            Getter pentru durata unei inchirieri
        :return: int
        """
        return self.__duration

    @duration.setter
    def duration(self, other_duration):
        """
            Setter pentru durata unei inchirieri
        :param other_duration: int - alta durata
        """
        self.__duration = other_duration

    def __eq__(self, other):
        """
            2 inchirieri sunt egale cand id-ul este egal
        :param other: int
        :return: True daca sunt egale, false altfel
        """
        return self.__id == other.id

    def __str__(self):
        """
            Rescrie formatul pentru string al unei carti
        :return: string - sirul formatat pentru afisarea unei carti
        """
        return str(self.__id) + ", " + str(self.__id_book) + ", " + self.__date + ", " + str(self.__duration)
