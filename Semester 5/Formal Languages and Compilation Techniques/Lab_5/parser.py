class Parser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.firsts = {}
        self.follows = {}
        self.table = {}

        self.__compute_firsts()
        self.__compute_follows()
        self.__build_table()

    def __compute_firsts(self):
        for production in self.grammar.productions:
            self.firsts[production.non_terminal] = set()

        while True:
            updated = False

            for production in self.grammar.productions:
                non_terminal = production.non_terminal
                for expansion in production.expansions:
                    first_symbol = expansion[0]

                    if first_symbol.islower():
                    # if first_symbol.isupper():
                        if not self.firsts[first_symbol].issubset(self.firsts[non_terminal]):
                            # non_terminal_firsts = (self.firsts[first_symbol] - {';'})
                            # for symbol in symbols[1:]:
                            #     if symbol.isupper():
                            #         non_terminal_firsts |= self.firsts[symbol]
                            #     else:
                            #         non_terminal_firsts.add(symbol)
                            # self.firsts[non_terminal] |= non_terminal_firsts
                            self.firsts[non_terminal] |= self.firsts[first_symbol]
                            updated = True
                    else:
                        if first_symbol not in self.firsts[non_terminal]:
                            self.firsts[non_terminal].add(first_symbol)
                            updated = True

            if not updated:
                break

    def __compute_first_symbol(self, expansion):
        first_symbol = expansion[0]

        if first_symbol.islower():
        # if first_symbol.isupper():
            return self.firsts[first_symbol]
        else:
            return {first_symbol}

    def __compute_follows(self):
        for production in self.grammar.productions:
            self.follows[production.non_terminal] = set()

        self.follows[self.grammar.start_symbol].add('$')

        while True:
            updated = False

            for production in self.grammar.productions:
                non_terminal = production.non_terminal
                for expansion in production.expansions:
                    for i in range(len(expansion)):
                        if expansion[i].islower():
                        # if expansion[i].isupper():
                            rest = expansion[i + 1:]

                            if not rest:
                                if not self.follows[non_terminal].issubset(self.follows[expansion[i]]):
                                    self.follows[expansion[i]] |= self.follows[non_terminal]
                                    updated = True
                            else:
                                first_set = self.__compute_first_symbol(rest)

                                to_check = first_set - {';'}
                                if not to_check.issubset(self.follows[expansion[i]]):
                                    self.follows[expansion[i]] |= to_check
                                    updated = True

                                if ';' in first_set or not rest:
                                    if not self.follows[non_terminal].issubset(self.follows[expansion[i]]):
                                        self.follows[expansion[i]] |= self.follows[non_terminal]
                                        updated = True

            if not updated:
                break

    def __build_table(self):
        for production in self.grammar.productions:
            non_terminal = production.non_terminal
            for expansion in production.expansions:
                first_set = self.__compute_first_symbol(expansion)

                first_set_without_epsilon = first_set - {';'}
                for terminal in first_set_without_epsilon:
                    if (non_terminal, terminal) in self.table:
                        print("Error: Grammar is not LL(1) - Conflict at First: ", non_terminal, terminal)
                        return
                    self.table[(non_terminal, terminal)] = [expansion, self.grammar.get_index(non_terminal, expansion)]

                if ';' in first_set or not production:
                    follow_set = self.follows[non_terminal]
                    for terminal in follow_set:
                        if (non_terminal, terminal) in self.table:
                            print("Error: Grammar is not LL(1) - Conflict at Follow: ", non_terminal, terminal)
                            return
                        self.table[(non_terminal, terminal)] = [expansion, self.grammar.get_index(non_terminal, expansion)]

    def parse(self, sequence):
        sequence = sequence.split('\n')
        sequence.append('$')
        stack = ['$', self.grammar.start_symbol]
        output = []
        output_sequence = []

        while True:
            if stack[-1] == '$' and sequence[0] == '$':
                output.append('ACCEPT')
                break

            if stack[-1] == sequence[0]:
                output.append('POP ' + stack[-1])
                stack.pop()
                sequence.pop(0)
                continue

            if stack[-1] in self.grammar.non_terminals:
                if (stack[-1], sequence[0]) in self.table:
                    output.append('PUSH ' + str(self.table[(stack[-1], sequence[0])][1]))
                    value = stack.pop()
                    output_sequence.append(self.table[(value, sequence[0])][1])
                    for symbol in reversed(self.table[(value, sequence[0])][0]):
                        if symbol != ';':
                            stack.append(symbol)
                else:
                    output.append('ERROR')
                    break
            else:
                output.append('ERROR')
                break

        return output, output_sequence

    def print_firsts(self):
        for non_terminal in self.firsts:
            print(non_terminal, self.firsts[non_terminal])

    def print_follows(self):
        for non_terminal in self.follows:
            print(non_terminal, self.follows[non_terminal])

    def print_table(self):
        for pair in self.table:
            print(pair, self.table[pair])
