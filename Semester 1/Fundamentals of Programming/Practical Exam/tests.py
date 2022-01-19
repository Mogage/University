import unittest
import datetime

from data import Events
from validator import Validator
from repository import FileRepository
from service import Service
import errors


class Tests(unittest.TestCase):
    def setUp(self):
        self.event_date = "01.01.2022"
        self.event_hour = "12:00"
        self.event_description = "Anul nou."

        self.event = Events(self.event_date, self.event_hour, self.event_description)

        self.validator = Validator()
        self.repository = FileRepository("tests.txt")
        self.service = Service(self.validator, self.repository)

    def tearDown(self):
        with open("tests.txt", "w"):
            pass

    def test_create_event(self):
        self.assertEqual(self.event.data, self.event_date)
        self.assertEqual(self.event.ora, self.event_hour)
        self.assertEqual(self.event.descriere, self.event_description)

    def test_rewrite_str(self):
        expected_string = "Data: 01.01.2022, Ora: 12:00, Descriere: Anul nou."
        self.assertEqual(str(self.event), expected_string)

    def test_validate_event_success(self):
        self.validator.validate_event(self.event)
        event = Events("25.02.22", "13:00", "")
        self.validator.validate_event(event)
        event = Events("26.04.22", "13:00", "")
        self.validator.validate_event(event)

    def test_validate_event_fail(self):
        event = Events("1.1.22", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("1-2-32", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("ab.cd.ef", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("32.01.22", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("29.02.22", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("31.04.22", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("31.13.22", "13:00", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("01.01.22", "1:0", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("01.01.22", "a-b", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("22.02.2020", "25:59", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("22.02.2020", "23:65", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)
        event = Events("22.02.2020", "ab:cd", "")
        self.assertRaises(errors.ValidatorError, self.validator.validate_event, event)

    def test_repository_event(self):
        self.repository.add_event(self.event)
        self.assertEqual(len(self.repository.all_events), 1)
        self.assertEqual(self.repository.all_events[0], self.event)

    def test_load_from_file(self):
        self.repository.add_event(self.event)
        new_repository = FileRepository("tests.txt")
        self.assertEqual(str(new_repository.all_events[0]), str(self.event))

    def test_service_add_event(self):
        self.service.add_event(self.event_date, self.event_hour, self.event_description)
        events = self.service.get_all_events()
        self.assertEqual(str(self.event), str(events[0]))

    def test_get_today_events(self):
        today_day = str(datetime.date.today().day)
        today_month = str(datetime.date.today().month)
        today_year = str(datetime.date.today().year)
        if len(today_day) == 1:
            today_day = '0' + today_day
        if len(today_month) == 1:
            today_month = '0' + today_month

        today_date = today_day + '.' + today_month + '.' + today_year
        self.service.add_event("01.01.2020", "15:20", "")
        self.service.add_event(today_date, "15:21", "")
        self.service.add_event(today_date, "15:20", "")
        self.service.add_event(today_date, "13:21", "")

        events = [Events(today_date, "13:21", ""), Events(today_date, "15:20", ""), Events(today_date, "15:21", "")]
        today_events = self.service.get_today_events()
        for i in range(3):
            self.assertEqual(str(today_events[i]), str(events[i]))

    def test_get_events_from_date(self):
        self.service.add_event("01.01.2020", "15:20", "")
        self.service.add_event("01.02.2020", "15:20", "")
        self.service.add_event("03.01.2020", "15:20", "")
        event = Events("01.01.2020", "15:20", "")
        events_from_date = self.service.get_events_from_date("01.01.2020")
        self.assertEqual(str(event), str(events_from_date[0]))

    def test_export_to_file(self):
        self.service.add_event("01.01.2020", "16:20", "da")
        self.service.add_event("01.02.2020", "15:21", "nu")
        self.service.add_event("03.01.2020", "15:20", "da")
        self.service.export_to_file("export_test.ts", "da")
        lines_exported = ["01.01.2020;16:20;da\n", "03.01.2020;15:20;da\n"]
        i = 0
        with open("export_test.txt", "r") as file:
            for line in file:
                self.assertEqual(str(line), lines_exported[i])
                i += 1
