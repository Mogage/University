from tree import Tree
from IOFactory import read_content_from_file


class CodeTransformer:
    def __init__(self):
        self.idTable = None
        self.__create_id_table()
        self.identifiers_automata = read_content_from_file('data/identifiers.txt')
        self.integer_numbers_automata = read_content_from_file('data/integerNumbers.txt')
        self.real_numbers_automata = read_content_from_file('data/realNumbers.txt')
        self.identifiers_tree = Tree()
        self.constants_tree = Tree()

    def __create_id_table(self):
        tableContents = [
            "ID", "CONST", "include", "using", "namespace", "std", "iostream", "math.h", "string.h",
            "const", "int", "float", "double", "void", "struct",
            "if", "while", "for", "cin", "cout", "return",
            "{", "}", "(", ")", ",", "<", ">", ";", "=", "!=", "==", "<=", ">=", "+", "-", "*", "/", "%", "<<", ">>",
            "#"]

        content_id = 0
        self.idTable = {}

        for content in tableContents:
            self.idTable[content] = content_id
            content_id += 1

    def transform_code(self, file_name, file_output=True):
        line_number = 0
        in_file_path = 'inputFiles/' + file_name
        out_file_path = 'outputFiles/' + file_name.replace('.cpp', '.txt')

        with open(in_file_path) as input_file:
            lines = input_file.readlines()
            for line in lines:
                line_number += 1
                if self.__work_line(line.replace('\n', ''), line_number) is False:
                    return

        self.identifiers_tree.print_tree()
        self.constants_tree.print_tree()

        if file_output:
            with open(out_file_path, 'w') as output_file:
                for line in lines:
                    self.__write_line(line.replace('\n', ''), output_file)

    def __write_line(self, line, output_file):
        line.strip()

        if line == '':
            return

        while len(line) > 0:
            if line[0] == ' ' or line[0] == '\t':
                line = line[1:]
                continue

            prefix, is_identifier = self.__work_prefix(line)
            if prefix not in self.idTable.keys():
                if is_identifier:
                    output_file.write(f'{prefix} -> 0 {self.identifiers_tree.get_index(prefix)}\n')
                else:
                    output_file.write(f'{prefix} -> 1 {self.constants_tree.get_index(prefix)}\n')
            else:
                output_file.write(f'{prefix} -> {self.idTable[prefix]} -\n')
            line = line[len(prefix):]

    def __work_line(self, line, line_number):
        line = line.strip()

        if line == '':
            return

        while len(line) > 0:
            if line[0] == ' ' or line[0] == '\t':
                line = line[1:]
                continue

            prefix, is_identifier = self.__work_prefix(line)
            if prefix not in self.idTable.keys():
                if len(prefix) >= 8:
                    print("Lexical error on line " + str(line_number) + " at position " + str(
                        len(prefix)) + ": " + prefix)
                    return False

                if is_identifier:
                    if self.identifiers_tree.search(prefix) is False:
                        self.identifiers_tree.insert(prefix)
                elif self.constants_tree.search(prefix) is False:
                    self.constants_tree.insert(prefix)

            line = line[len(prefix):]

    def __work_prefix(self, line):
        longest_identifier = self.identifiers_automata.get_longest_prefix(line)
        longest_integer_number = self.integer_numbers_automata.get_longest_prefix(line)
        longest_real_number = self.real_numbers_automata.get_longest_prefix(line)

        if longest_identifier != '':
            return longest_identifier, True
        elif longest_real_number != '':
            return longest_real_number, False
        elif longest_integer_number != '':
            return longest_integer_number, False

        if len(line) > 1:
            if line[0] == '>' or line[0] == '<' or line[0] == '=' or line[0] == '!':
                if line[1] == '=':
                    return line[0:2], False

            if line[0] == '>' and line[1] == '>' or line[0] == '<' and line[1] == '<':
                return line[0:2], False

        return line[0], False
