class Events:
    """
        Clasa pentru entitatea de tip eveniment
    """
    def __init__(self, data, ora, descriere):
        """
            Initializarea atributelor unui eveniment
        :param data: string, o data in formatul dd.mm.yyyy
        :param ora: string, o ora in formatul hh:mm
        :param descriere: string
        """
        self.__data = data
        self.__ora = ora
        self.__descriere = descriere

    @property
    def data(self):
        """
            Getter pentru data eveniment
        :return: string, data evenimentului
        """
        return self.__data

    @property
    def ora(self):
        """
            Getter pentru ora eveniment
        :return: string, ora evenimentului
        """
        return self.__ora

    @property
    def descriere(self):
        """
            Getter pentru descriere eveniment
        :return: string, descrierea evenimentului
        """
        return self.__descriere

    def __str__(self):
        """
            Suprascrie operatia str() pentru o entitate de tip event
        :return: string, in formatul "Data: {data}, "Ora: {ora}, "Descriere: {descriere}"
        """
        return "Data: " + self.__data + ", Ora: " + self.__ora + ", Descriere: " + self.__descriere
