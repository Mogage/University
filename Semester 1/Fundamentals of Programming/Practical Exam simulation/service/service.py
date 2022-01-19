class BookService:
    """
        Clasa pentru efectuare operatiilor cu carti
    """
    def __init__(self, books_repository):
        """
            Initializarea atributelor unui serviciu de carti
        :param books_repository: BooksRepository
        """
        self.__books_repository = books_repository

    def requirement_1(self, user_input):
        """
            Cauta cartile a caror titlu se termina cu user_input si returneaza o lista cu aceste carti
        :param user_input: string
        :return: list[] - lista cu cartile care au proprietatea ceruta
        """
        filtered_list = []
        for _book in self.__books_repository.books:
            if _book.title.endswith(user_input):
                filtered_list.append(_book)

        filtered_list.sort(key=lambda book: book.year)

        return filtered_list


class RentService:
    """
        Clasa pentru efectuarea operatiilor cu inchirieri
    """
    def __init__(self, rent_repository):
        """
            Initializarea atributelor unui serviciu de inchirieri
        :param rent_repository: RentRepository
        """
        self.__rent_repository = rent_repository

    def requirement_2(self, user_input):
        """
            Cauta inchirierile a caror durata este egala cu user_input
        :param user_input: int > 0
        :return: list[] - lista cu inchirierile care au proprietatea ceruta
        """
        filtered_list = []
        for _rent in self.__rent_repository.rents:
            if _rent.duration == user_input:
                filtered_list.append(_rent)

        return filtered_list
