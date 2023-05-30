import unittest

class TestsAPI(unittest.TestCase):

    def testDefault(self):
        self.assertEqual(True, True)

    def testFail(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()