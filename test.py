import unittest
import yel2

dt = None

class TestCode(unittest.TestCase):
    def setUp(self):
        self.dt=[1,2,3]

    def test_denom_a_func(self):
        self.assertequal(denom_a_func(0),6)

unittest.main(exit = False)
