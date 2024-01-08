from utils import list_to_string


class _Production:
    def __init__(self):
        self.non_terminal = None
        self.expansions = []

    def __str__(self):
        return self.non_terminal + ' -> ' + list_to_string(self.expansions)


class Grammar:
    def __init__(self):
        self.productions = []
        self.non_terminals = set()
        self.start_symbol = None

    def create_grammar(self, file_name):
        with open(file_name, 'r') as file:
            self.start_symbol = file.readline().strip()
            for line in file:
                line = line.strip()
                if line == '':
                    continue

                left_side = line.split('->')[0].strip()
                self.non_terminals.add(left_side)
                right_side = line.split('->')[1].strip()

                production = _Production()
                production.non_terminal = left_side
                production.expansions = right_side.split('|')
                for i in range(len(production.expansions)):
                    production.expansions[i] = production.expansions[i].strip().split(' ')
                self.productions.append(production)

    def __str__(self):
        string = ''
        count = 1
        for production in self.productions:
            for expansion in production.expansions:
                string += str(production.non_terminal) + ' -> ' + list_to_string(expansion) + ' ' + str(count) + '\n'
                count += 1
        return string

    def get_by_index(self, index):
        count = 0
        for production in self.productions:
            for expansion in production.expansions:
                if count == index:
                    return [production.non_terminal, expansion]
                count += 1

    def get_index(self, non_terminal, expansion):
        count = 1
        for production in self.productions:
            if production.non_terminal == non_terminal:
                for exp in production.expansions:
                    if exp == expansion:
                        return count
                    count += 1
            else:
                count += len(production.expansions)
        return -1
