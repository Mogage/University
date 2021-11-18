"""
    Author: Mogage Nicolae
    Creation date: 6 nov 2021
    Modul pentru stocarea entitatilor client
"""
from domain.data import Clients
from error.errors import ClientRepositoryError

class ClientsRepository:
    """
        Clasa ce se ocupa cu stocarea clientilor
    """
    def __init__(self):
        """
            Functie de initializare a unei liste entitati de tip client
        """
        self.__clients = []

    @property
    def clients(self):
        """
            Functie de getter pentru lista de entitati
        :return: __clients
        """
        return self.__clients

    def find_client_by_id(self, id):
        """
            Cauta un client cu id-ul id
        :param id: id > 0
        :return: entitatea de tip client cu id-ul id
        :raise: Id client inexistent. daca nu exista un client cu acest id
        """
        for _client in self.__clients:
            if _client.id == id:
                return _client
        raise ClientRepositoryError("Id client inexistent.")

    def find_client_by_cnp(self, cnp):
        """
            Cauta un client cu cnp-ul cnp
        :param cnp: int cu exact 13 cifre
        :return: entitate de tip client cu cnp-ul cnp
        :raise: Cnp client inexistent. daca nu exista un client cu acest cnp.
        """
        for _client in self.__clients:
            if _client.cnp == cnp:
                return _client
        raise ClientRepositoryError("Cnp client inexistent.")

    def add_client(self, client):
        """
            Adauga un client inexistent in repertoriu, un client exista deja daca este alt client cu acelasi id sau cnp
        :param client: un client
        :return: -
        :raises: Exista deja un client cu acest id. daca s-a gasit un client cu acelasi id
                 Exista deja un client cu acest cnp. daca s-a gasit un client cu acelasi cnp
        """
        for _client in self.__clients:
            if _client.id == client.id:
                raise ClientRepositoryError("Exista deja un client cu acest id.")
            
        for _client in self.__clients:
            if _client.cnp == client.cnp:
                raise ClientRepositoryError("Exista deja un client cu acest cnp.")

        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients and client > self.__clients[index]:
            index = index + 1

        self.__clients.insert(index, client)

    def update_name(self, id, name):
        """
            Actualizeaza numele unui client cu id-ul id
        :param id: int > 0
        :param name: not empty string
        :return: -
        :raise: Id client inexistent. daca nu exista un client cu acest id.
        """
        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients:
            if self.__clients[index].id == id:
                self.__clients[index].name = name
                return
            index = index + 1
        raise ClientRepositoryError("Id client inexistent.")

    def delete_by_id(self, id):
        """
            Sterge un client cu id-ul id
        :param id: int > 0
        :return: -
        :raise: Id client inexistent. daca nu exista un client cu acest id.
        """
        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients:
            if self.__clients[index].id == id:
                self.__clients.pop(index)
                return
            index = index + 1

        raise ClientRepositoryError("Id client inexistent.")

    def delete_by_name(self, name):
        """
            Sterge clientii cu numele name
        :param name: not empty string
        :return: -
        :raise: Nume client inexistent. daca nu exista clienti cu acest nume
        """
        index = 0
        number_of_clients = len(self.__clients)
        client_is_deleted = False
        while index < number_of_clients:
            if self.__clients[index].name == name:
                self.__clients.pop(index)
                index = index - 1
                number_of_clients = number_of_clients - 1
                client_is_deleted = True
            index = index + 1

        if not client_is_deleted:
            raise ClientRepositoryError("Nume client inexistent.")

    def delete_by_cnp(self, cnp):
        """
            Sterge clientul cu cnp-ul cnp
        :param cnp: int cu exact 13 cifre
        :return: -
        :raise: Cnp client inexistent. daca nu exista un client cu acest cnp
        """
        index = 0
        number_of_clients = len(self.__clients)
        while index < number_of_clients:
            if self.__clients[index].cnp == cnp:
                self.__clients.pop(index)
                return
            index = index + 1

        raise ClientRepositoryError("Cnp client inexistent.")

    def print_all_clients(self):
        """
            Afiseaza toti clientii
        :return: -
        """
        for _client in self.__clients:
            print(str(_client))

    def print_clients_id(self, id):
        """
            Afiseaza clienul cu id-ul id
        :param id: int > 0
        :return: -
        :raise: Id carte inexistent. daca nu exista un client cu acest id
        """
        for _client in self.__clients:
            if _client.id == id:
                print(str(_client))
                return
        raise ClientRepositoryError("Id carte inexistent.")

    def print_clients_name(self, name):
        """
            Afiseaza clientii cu numele name
        :param name: not empty string
        :return: -
        :raise: Nume client inexistent. daca nu exista clienti cu acest nume
        """
        client_exists = False
        for _client in self.__clients:
            if _client.name == name:
                print(str(_client))
                client_exists = True

        if not client_exists:
            raise ClientRepositoryError("Nume client inexistent.")

    def print_clients_cnp(self, cnp):
        """
            Afiseaza clientul cu cnp-ul cnp
        :param cnp: not empty string
        :return: -
        :raise: Cnp client inexistent. daca nu exista un client cu acest cnp
        """
        for _client in self.__clients:
            if _client.cnp == cnp:
                print(str(_client))
                return
        raise ClientRepositoryError("Cnp client inexistent.")

    def filter_name_by_prefix(self, prefix):
        """
            Se returneaza o lista cu clientii a coror nume incep cu prefixul prefix
        :param prefix: string
        :return: lista filtrata
        :raise: Nu exista niciun client a carui nume sa inceapa cu acest prefix. daca nu se gaseste un client care sa inceapa cu acel prefix
        """
        filtered_list = []
        for _client in self.__clients:
            if _client.name.startswith(prefix):
                filtered_list.append(_client)

        if len(filtered_list):
            return filtered_list

        raise ClientRepositoryError("Nu exista niciun client a carui nume sa inceapa cu acest prefix.")

    def print_filter_name_by_prefix(self, prefix):
        """
            Afiseaza clientii a caror nume incep cu prefixul prefix
        :param prefix: string
        """
        filtered_list = self.filter_name_by_prefix(prefix)
        for _filtered_client in filtered_list:
            print(str(_filtered_client))

    def __len__(self):
        """
            Lungimea unui repozitoriu este egala cu numarul de clienti inregistrati
        :return: int - numarul de clienti inregistrati
        """
        return len(self.__clients)

class ClientsFileRepository(ClientsRepository):
    """
        Clasa pentru stocarea in fisiere a clientilor
    """
    def __init__(self, file_name):
        """
            Initializarea datelor pentru stocarea in fisiere
        :param file_name: nume fisier - string
        """
        self.__file_name = file_name
        ClientsRepository.__init__(self)
        self.__load_from_file()

    @staticmethod
    def __create_client_from_file(line):
        """
            Creeaza un client din string-ul line in care atributele sunt despartite prin ;
        :param line: o linie formatata din fisier - string
        :return: un client
        """
        fields = line.split(";")
        fields[2] = fields[2].removesuffix("\n")
        client = Clients(int(fields[0]), fields[1], fields[2])
        return client

    def __load_from_file(self):
        """
            Incarca toate datele din fisier
        """
        with open(self.__file_name) as client_file:
            for line in client_file:
                client = self.__create_client_from_file(line)
                ClientsRepository.add_client(self, client)

    @staticmethod
    def __create_line(client):
        """
            Formateaza un client pentru a fi stocat in fisier
        :param client: un client
        :return: un string cu atributele separate prin ;
        """
        return str(client.id) + ";" + client.name + ";" + str(client.cnp)

    @staticmethod
    def __add_to_file(line, client_file):
        """
            Scrie in fisier clientul formatat pentru o linie
        :param line: client formatat sub forma de linie - string
        :param client_file: fisierul in care se vor stoca
        """
        client_file.write(line)
        client_file.write("\n")

    def rewrite_file(self):
        """
            Scrie in fisier toti clientii
        """
        with open(self.__file_name, "w") as client_file:
            for _client in self.clients:
                line = self.__create_line(_client)
                self.__add_to_file(line, client_file)
