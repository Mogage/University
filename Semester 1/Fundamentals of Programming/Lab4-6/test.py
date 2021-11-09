import expenses
import service
import undo


def test_create_expense():
    expense = expenses.create_expense(12, 250, "mancare")
    assert expenses.get_expense_day(expense) == 12
    assert expenses.get_expense_value(expense) == 250
    assert expenses.get_expense_type(expense) == "mancare"

def test_check_type():
    expense = expenses.create_expense(12, 250, "mancare")
    assert expenses.check_type(expenses.get_expense_type(expense)) == True
    expense = expenses.create_expense(12, 250, "apa")
    assert expenses.check_type(expenses.get_expense_type(expense)) == False
    expense = expenses.create_expense(12, 250, "Mancare")
    assert expenses.check_type(expenses.get_expense_type(expense)) == True

def test_check_expense():
    expense = expenses.create_expense(0, 10000, "altele")
    try:
        expenses.check_expense(expense)
        assert False
    except Exception as ex:
        assert str(ex) == "Ziua introdusa este invalida.\nSuma introdusa este invalida.\n"
    expense = expenses.create_expense(0, 0, "")
    try:
        expenses.check_expense(expense)
        assert False
    except Exception as ex:
        assert str(ex) == "Ziua introdusa este invalida.\nTipul introdus este invalid.\nSuma introdusa este invalida.\n"
    expense = expenses.create_expense(12, 250, "mancare")
    expenses.check_expense(expense)
    expense = expenses.create_expense(32, 250, "altele")
    try:
        expenses.check_expense(expense)
        assert False
    except Exception as ex:
        assert str(ex) == "Ziua introdusa este invalida.\n"
    expense = expenses.create_expense(32, 250, "apa")
    try:
        expenses.check_expense(expense)
        assert False
    except Exception as ex:
        assert str(ex) == "Ziua introdusa este invalida.\nTipul introdus este invalid.\n"
    expense = expenses.create_expense(32, -256, "apa")
    try:
        expenses.check_expense(expense)
        assert False
    except Exception as ex:
        assert str(ex) == "Ziua introdusa este invalida.\nTipul introdus este invalid.\nSuma introdusa este invalida.\n"

def test_equal_expenses():
    expense_1 = expenses.create_expense(15, 200, "telefon")
    expense_2 = expenses.create_expense(15, 205, "telefon")
    expense_3 = expenses.create_expense(15, 20, "imbracaminte")
    assert expenses.equal_expenses(expense_1, expense_2) == True
    assert expenses.equal_expenses(expense_1, expense_3) == False
    assert expenses.equal_expenses(expense_2, expense_3) == False

def test_add_expense_to_list():
    list = []
    expense = expenses.create_expense(15, 200, "telefon")
    assert service.srv_empty_list(list) == True
    expenses.add_expense_to_list(list, expense)
    assert len(list) == 1
    assert expenses.get_expense_day(list[0]) == 15
    assert expenses.get_expense_value(list[0]) == 200
    assert expenses.get_expense_type(list[0]) == "telefon"
    expense_2 = expenses.create_expense(15, 205, "telefon")
    assert len(list) == 1
    try:
        expenses.add_expense_to_list(list, expense_2)
        assert False
    except Exception as ex:
        assert str(ex) == "Cheltuiala existenta.\n"
    expense_3 = expenses.create_expense(17, 205, "telefon")
    expenses.add_expense_to_list(list, expense_3)
    assert len(list) == 2
    assert expenses.get_expense_day(list[1]) == 17
    assert expenses.get_expense_value(list[1]) == 205
    assert expenses.get_expense_type(list[1]) == "telefon"

def test_srv_add_expense_to_list():
    list = []
    assert service.srv_empty_list(list) == True
    service.srv_add_expense_to_list(list, 15, 10, "telefon")
    try:
        service.srv_add_expense_to_list(list, -2, 19, "")
        assert False
    except Exception as ex:
        assert str(ex) == "Ziua introdusa este invalida.\nTipul introdus este invalid.\n"
    try:
        service.srv_add_expense_to_list(list, 15, 20, "telefon")
        assert False
    except Exception as ex:
        assert str(ex) == "Cheltuiala existenta.\n"

def test_srv_update_expense():
    list = []
    assert service.srv_empty_list(list) == True
    service.srv_add_expense_to_list(list, 15, 20, "intretinere")
    service.srv_update_expense(list, 15, 35, "intretinere")
    assert expenses.get_expense_value(list[0]) == 35
    try:
        service.srv_update_expense(list, 14, 20, "intretinere")
        assert False
    except Exception as ex:
        assert str(ex) == "Cheltuiala nu exista.\n"
    try:
        service.srv_update_expense(list, 15, 50, "altele")
        assert False
    except Exception as ex:
        assert str(ex) == "Cheltuiala nu exista.\n"

def test_srv_delete_expense_by_day():
    list = []
    service.srv_add_expense_to_list(list, 15, 10, "telefon")
    assert service.srv_delete_expense_by_day(list, 17) == "Nu s-a efectuat nicio cheltuiala in aceasta zi."
    assert len(list) == 1
    assert service.srv_delete_expense_by_day(list, 15) == "Cheltuiala stearsa cu succes."
    assert len(list) == 0
    service.srv_add_expense_to_list(list, 20, 200, "mancare")
    service.srv_add_expense_to_list(list, 20, 20, "intretinere")
    service.srv_add_expense_to_list(list, 21, 200, "mancare")
    assert service.srv_delete_expense_by_day(list, 20) == "Cheltuieli sterse cu succes."
    assert len(list) == 1

def test_srv_delete_expense_between_days():
    list = []
    service.srv_add_expense_to_list(list, 25, 175, "intretinere")
    assert service.srv_delete_expense_between_days(list, 19, 22) == "Nu s-a efectuat nicio cheltuiala in acest interval de timp."
    assert len(list) == 1
    service.srv_add_expense_to_list(list, 20, 200, "altele")
    assert service.srv_delete_expense_between_days(list, 20, 20) == "Cheltuieli sterse cu succes."
    assert len(list) == 1
    service.srv_add_expense_to_list(list, 7, 250, "mancare")
    assert service.srv_delete_expense_between_days(list, 5, 28) == "Cheltuieli sterse cu succes."
    assert len(list) == 0

def test_srv_delete_expense_by_type():
    list = []
    assert service.srv_empty_list(list) == True
    service.srv_add_expense_to_list(list, 15, 10, "telefon")
    assert service.srv_delete_expense_by_type(list, "altele") == "Nu s-a efectuat nicio cheltuiala de acest tip."
    assert service.srv_delete_expense_by_type(list, "telefon") == "Cheltuiala stearsa cu succes."
    assert service.srv_empty_list(list) == True
    service.srv_add_expense_to_list(list, 15, 10, "telefon")
    service.srv_add_expense_to_list(list, 14, 25, "telefon")
    assert len(list) == 2
    assert service.srv_empty_list(list) == False
    assert service.srv_delete_expense_by_type(list, "telefon") == "Cheltuieli sterse cu succes."
    assert service.srv_empty_list(list) == True

def test_srv_print_expenses():
    list = []
    assert service.srv_print_expenses(list) == ""
    service.srv_add_expense_to_list(list, 20, 15, "mancare")
    assert service.srv_print_expenses(list) == "Ziua: 20 Tip: mancare Suma: 15."
    service.srv_add_expense_to_list(list, 15, 20, "telefon")
    assert service.srv_print_expenses(list) == "Ziua: 15 Tip: telefon Suma: 20.\nZiua: 20 Tip: mancare Suma: 15."

def test_srv_print_expense_greater_than_value():
    list = []
    service.srv_add_expense_to_list(list, 20, 15, "mancare")
    assert service.srv_print_expense_greater_than_value(list, 20) == "Nu s-a efectuat nicio cheltuiala mai mare decat aceasta suma."
    assert service.srv_print_expense_greater_than_value(list, 15) == "Nu s-a efectuat nicio cheltuiala mai mare decat aceasta suma."
    assert service.srv_print_expense_greater_than_value(list, 10) == "Ziua: 20 Tip: mancare Suma: 15."
    service.srv_add_expense_to_list(list, 19, 19, "altele")
    assert service.srv_print_expense_greater_than_value(list, 5) == "Ziua: 19 Tip: altele Suma: 19.\nZiua: 20 Tip: mancare Suma: 15."

def test_srv_print_expense_before_day_lower_value():
    list = []
    service.srv_add_expense_to_list(list, 15, 20, "mancare")
    assert service.srv_print_expense_before_day_lower_value(list, 10, 50) == "Nu s-a efectuat nicio cheltuiala inainte de aceasta zi."
    assert service.srv_print_expense_before_day_lower_value(list, 20, 10) == "Nu s-a efectuat nicio cheltuiala mai mica decat aceasta suma."
    assert service.srv_print_expense_before_day_lower_value(list, 20, 25) == "Ziua: 15 Tip: mancare Suma: 20."
    service.srv_add_expense_to_list(list, 10, 15, "altele")
    assert service.srv_print_expense_before_day_lower_value(list, 20, 25) == "Ziua: 10 Tip: altele Suma: 15.\nZiua: 15 Tip: mancare Suma: 20."

def test_srv_print_expense_by_type():
    list = []
    assert service.srv_empty_list(list) == True
    service.srv_add_expense_to_list(list, 15, 10, "telefon")
    assert service.srv_print_expense_by_type(list, "telefon") == "Ziua: 15 Tip: telefon Suma: 10."
    assert service.srv_print_expense_by_type(list, "altele") == "Nu s-a efectuat nicio cheltuiala de acest tip."
    service.srv_add_expense_to_list(list, 14, 20, "telefon")
    service.srv_add_expense_to_list(list, 15, 15, "imbracaminte")
    assert service.srv_print_expense_by_type(list, "telefon") == "Ziua: 14 Tip: telefon Suma: 20.\nZiua: 15 Tip: telefon Suma: 10."

def test_srv_print_total_values_by_type():
    list = []
    service.srv_add_expense_to_list(list, 25, 500, "intretinere")
    assert service.srv_print_total_values_by_type(list, "intretinere") == "Suma totala pentru tipul de cheltuieli selectat este de 500 de lei."
    assert service.srv_print_total_values_by_type(list, "altele") == "Nu s-a efectuat nicio cheltuiala de acest tip."
    service.srv_add_expense_to_list(list, 20, 19, "telefon")
    service.srv_add_expense_to_list(list, 19, 1, "altele")
    service.srv_add_expense_to_list(list, 15, 40, "intretinere")
    assert service.srv_print_total_values_by_type(list, "telefon") == "Suma totala pentru tipul de cheltuieli selectat este de 19 lei."
    assert service.srv_print_total_values_by_type(list, "altele") == "Suma totala pentru tipul de cheltuieli selectat este de 1 leu."
    assert service.srv_print_total_values_by_type(list, "intretinere") == "Suma totala pentru tipul de cheltuieli selectat este de 540 de lei."

def test_srv_find_max_value_expense():
    list = []
    service.srv_add_expense_to_list(list, 15, 15, "telefon")
    assert service.srv_find_max_value_expense(list) == [15, 15]
    service.srv_add_expense_to_list(list, 12, 10, "altele")
    assert service.srv_find_max_value_expense(list) == [15, 15]
    service.srv_add_expense_to_list(list, 12, 15, "telefon")
    assert service.srv_find_max_value_expense(list) == [12, 25]

def test_srv_print_expense_by_value():
    list = []
    service.srv_add_expense_to_list(list, 15, 10, "telefon")
    assert service.srv_print_expense_by_value(list, 10) == "Ziua: 15 Tip: telefon Suma: 10."
    assert service.srv_print_expense_by_value(list, 20) == "Nu s-a efectuat nicio cheltuiala cu aceasta suma."
    service.srv_add_expense_to_list(list, 14, 10, "telefon")
    service.srv_add_expense_to_list(list, 15, 10, "imbracaminte")
    assert service.srv_print_expense_by_value(list, 10) == "Ziua: 14 Tip: telefon Suma: 10.\nZiua: 15 Tip: telefon Suma: 10.\nZiua: 15 Tip: imbracaminte Suma: 10."

def test_srv_expense_filter_by_type():
    list = []
    service.srv_add_expense_to_list(list, 20, 99, "mancare")
    list = service.srv_expense_filter_by_type(list, "mancare")
    assert len(list) == 0
    assert service.srv_print_expenses(list) == ""
    service.srv_add_expense_to_list(list, 20, 99, "mancare")
    service.srv_add_expense_to_list(list, 15, 100, "telefon")
    list = service.srv_expense_filter_by_type(list, "telefon")
    assert len(list) == 1
    assert service.srv_print_expenses(list) == "Ziua: 20 Tip: mancare Suma: 99."

def test_srv_expense_filter_lower_value():
    list = []
    service.srv_add_expense_to_list(list, 28, 105, "altele")
    list = service.srv_expense_filter_lower_value(list, 50)
    assert len(list) == 1
    list = service.srv_expense_filter_lower_value(list, 150)
    assert len(list) == 0
    assert service.srv_print_expenses(list) == ""
    service.srv_add_expense_to_list(list, 20, 104, "mancare")
    service.srv_add_expense_to_list(list, 20, 20, "altele")
    service.srv_add_expense_to_list(list, 19, 109, "telefon")
    list = service.srv_expense_filter_lower_value(list, 104)
    assert len(list) == 2
    assert service.srv_print_expenses(list) == "Ziua: 19 Tip: telefon Suma: 109.\nZiua: 20 Tip: mancare Suma: 104."

def test_update():
    list = []
    undo_stack = []
    service.srv_add_expense_to_list(list, 25, 105, "altele")
    service.srv_update_expense(list, 25, 100, "altele", undo_stack)
    # assert undo.get_value(undo_stack[-1]) == {'day': 25, 'type': "altele", 'value': 105}
    assert undo.get_value(undo_stack[-1]) == [25, "altele", 105]
    assert service.srv_print_expenses(list) == "Ziua: 25 Tip: altele Suma: 100."
    undo.update(list, undo_stack[-1])
    assert service.srv_print_expenses(list) == "Ziua: 25 Tip: altele Suma: 105."

def test_delete():
    list = []
    stack = []
    service.srv_add_expense_to_list(list, 10, 200, "telefon")
    service.srv_add_expense_to_list(list, 20, 10, "telefon")
    service.srv_delete_expense_by_day(list, 10, stack)
    # assert stack[-1]["deleted_expenses"] == [{"expense": {"day": 10, "type": "telefon", "value": 200}, "position": 0}]
    assert stack[-1]["deleted_expenses"] == [{"expense": [10, "telefon", 200], "position": 0}]
    undo.delete(list, stack[-1])
    stack.pop()
    assert service.srv_print_expenses(list) == "Ziua: 10 Tip: telefon Suma: 200.\nZiua: 20 Tip: telefon Suma: 10."
    service.srv_add_expense_to_list(list, 10, 50, "altele")
    service.srv_delete_expense_by_day(list, 10, stack)
    assert service.srv_print_expenses(list) == "Ziua: 20 Tip: telefon Suma: 10."
    # assert stack[-1]["deleted_expenses"] == [{"expense": {"day": 10, "type": "telefon", "value": 200}, "position": 0}, {"expense": {"day": 10, "type": "altele", "value": 50}, "position": 0}]
    assert stack[-1]["deleted_expenses"] == [{"expense": [10, "telefon", 200], "position": 0}, {"expense": [10, "altele", 50], "position": 0}]
    undo.delete(list, stack[-1])
    assert service.srv_print_expenses(list) == "Ziua: 10 Tip: telefon Suma: 200.\nZiua: 10 Tip: altele Suma: 50.\nZiua: 20 Tip: telefon Suma: 10."

def run_tests():
    test_create_expense()
    test_check_expense()
    test_check_type()
    test_equal_expenses()
    test_add_expense_to_list()
    test_srv_add_expense_to_list()
    test_srv_update_expense()
    test_srv_delete_expense_by_type()
    test_srv_delete_expense_by_day()
    test_srv_delete_expense_between_days()
    test_srv_print_expenses()
    test_srv_print_expense_greater_than_value()
    test_srv_print_expense_before_day_lower_value()
    test_srv_print_expense_by_type()
    test_srv_print_total_values_by_type()
    test_srv_find_max_value_expense()
    test_srv_print_expense_by_value()
    test_srv_expense_filter_by_type()
    test_srv_expense_filter_lower_value()
    test_update()
    test_delete()
