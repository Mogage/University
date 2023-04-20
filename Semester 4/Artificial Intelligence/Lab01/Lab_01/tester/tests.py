from problems import Laborator_01
from math import sqrt

class Tester:
    def _test_01(self):
        assert(Laborator_01.problem_01('ana ana Ana ana ana') == 'ana')
        assert(Laborator_01.problem_01('Ana are mere rosii si galbene') == 'si')
        assert(Laborator_01.problem_01('Abc aBc abC') == 'abC')

    def _test_02(self):
        assert(Laborator_01.problem_02(1,5,4,1) == 5)
        assert(Laborator_01.problem_02(1,1,1,1) == 0)
        assert(Laborator_01.problem_02(1,1,5,5) == sqrt(32))

    def _test_03(self):
        assert(Laborator_01.problem_03([[1,0,2,0,3]], [[1,2,0,3,1]]) == 4)
        assert(Laborator_01.problem_03([[1,0,2,0,3],[1,0,2,0,3]], [[1,2,0,3,1],[1,2,0,3,1]]) == 8)
        assert(Laborator_01.problem_03([[1,0,2,0,3,1,0,2,0,3,1,0,2,0,3],
                                        [1,0,2,0,3,1,0,2,0,3,1,0,2,0,3],
                                        [1,0,2,0,3,1,0,2,0,3,1,0,2,0,3]], 
                                        [[1,2,0,3,1,1,2,0,3,1,1,2,0,3,1],
                                         [1,2,0,3,1,1,2,0,3,1,1,2,0,3,1],
                                         [1,2,0,3,1,1,2,0,3,1,1,2,0,3,1]]) 
                                        == 36)

    def _test_04(self):
        assert(Laborator_01.problem_04('ana are ana are mere rosii ana') == {'mere', 'rosii'})
        assert(Laborator_01.problem_04('ana are mere rosii') == {'ana', 'are', 'mere', 'rosii'})
        assert(Laborator_01.problem_04('ana are mere rosii ana are mere rosii') == set())

    def _test_05(self):
        assert(Laborator_01.problem_05([1,2,3,4,2]) == 2)
        assert(Laborator_01.problem_05([1,2,3,4,5,6,7,8,9,10,11,12,12]) == 12)
        assert(Laborator_01.problem_05([1,1]) == 1)
    
    def _test_06(self):
        assert(Laborator_01.problem_06([2,8,7,2,2,5,2,3,1,2,2]) == 2)
        assert(Laborator_01.problem_06([2,1,2]) == 2)
        assert(Laborator_01.problem_06([1,3,2,2]) == None)

    def _test_07(self):
        assert(Laborator_01.problem_07([7,4,6,3,9,1], 2) == 7)
        assert(Laborator_01.problem_07([7,4,6,3,9,1], 6) == 1)
        assert(Laborator_01.problem_07([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 1) == 15)
        assert(Laborator_01.problem_07([7,7,7,7,7,7,7,7,7,7], 2) == 7)
        assert(Laborator_01.problem_07([7,7,7,7,7,7,7,7,7,7], 7) == 7)
    
    def _test_08(self):
        assert(Laborator_01.problem_08(4) == ['1', '10', '11', '100'])
        assert(Laborator_01.problem_08(25) == ['1', '10', '11', '100', '101', '110', '111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111', '11000', '11001'])
        assert(Laborator_01.problem_08(1) == ['1'])

    def _test_09(self):
        matrix = [[0, 2, 5, 4, 1],
                [4, 8, 2, 3, 7],
                [6, 3, 4, 6, 2],
                [7, 3, 1, 8, 3],
                [1, 5, 7, 9, 4]]
        assert(Laborator_01.problem_09(matrix, 1,1,3,3) == 38)
        assert(Laborator_01.problem_09(matrix, 2,2,4,4) == 44)
        assert(Laborator_01.problem_09(matrix, 4,4,2,2) == 44)
        assert(Laborator_01.problem_09(matrix, 0,4,4,0) == 105)

    def _test_10(self):
        matrix = [[0,0,0,1,1],
                [0,1,1,1,1],
                [0,0,1,1,1]]
        assert(Laborator_01.problem_10(matrix) == 2)
        matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        assert(Laborator_01.problem_10(matrix) == 12)
        matrix = [[0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]]
        assert(Laborator_01.problem_10(matrix) == 0)

    def run(self):
        self._test_01()
        self._test_02()
        self._test_03()
        self._test_04()
        self._test_05()
        self._test_06()
        self._test_07()
        self._test_08()
        self._test_09()
        self._test_10()
        print("Test passed")