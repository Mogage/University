from errors import ValidatorError


class Validator:
    """
        Clasa pentru verificarea evenimentelor si atributelor ei
    """
    @staticmethod
    def validate_date(date):
        """
            Verifica daca o data respecta formatul dd.mm.yyyy si daca este corecta
        :param date: string, o data
        :return: True, daca data este corecta, False altfel
        """
        month_31_days = [1, 3, 5, 7, 8, 10, 12]
        month_30_days = [4, 6, 8, 11]
        fields = date.split('.')
        if len(fields) != 3:
            return False

        if len(fields[0]) != 2 or len(fields[1]) != 2:
            return False

        try:
            days = int(fields[0])
            month = int(fields[1])
            int(fields[2])
        except ValueError:
            return False

        if month < 1 or month > 12:
            return False

        if month == 2:
            if 1 <= days <= 28:
                return True
            return False

        if month in month_30_days:
            if 1 <= days <= 30:
                return True
            return False

        if month in month_31_days:
            if 1 <= days <= 31:
                return True
            return False

    @staticmethod
    def __validate_hour(hour):
        """
            Verifica daca o ora respecta formatul "hh:mm" si daca este corecta
        :param hour: string, o ora
        :return: True, daca ora este corecta, False altfel
        """
        fields = hour.split(':')
        if len(fields) != 2:
            return False

        if len(fields[0]) != 2 or len(fields[1]) != 2:
            return False

        try:
            hours = int(fields[0])
            minutes = int(fields[1])
        except ValueError:
            return False

        if hours < 0 or hours > 24:
            return False

        if minutes < 0 or minutes > 59:
            return False

        return True

    def validate_event(self, event):
        """
            Verifica daca un eveniment este corect
        :param event: entitate de tip Events
        :raise: ValidationError(errors), if len(errors) > 0; errors - string cu erorile care exista
        """
        errors = ""
        if not self.validate_date(event.data):
            errors += "Data evenimentului trebuie sa fie in formatul dd.mm.yyyy.\n"
        if not self.__validate_hour(event.ora):
            errors += "Ora evenimentului trebuie sa fie in formatul hh:mm.\n"

        if len(errors) > 0:
            raise ValidatorError(errors)
