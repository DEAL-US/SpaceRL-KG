import unittest

class TestsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # THIS METHOD RUNS ONCE BEFORE EVERY TEST IS PERFORMED.
        self.var = "TEST"

    def testDefault(self):
        self.assertEqual(True, True)

    def testFail(self):
        self.assertEqual(True, True)

    def testConfig(self):
        self.assertEqual(True, True)

    def testPie(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()