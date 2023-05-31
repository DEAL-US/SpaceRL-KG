import unittest

class TestsAPI(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        print("HEWWO")
        super(TestsAPI, self).__init__(*args, **kwargs)

    def testDefault(self):
        self.assertEqual(True, True)

    def testFail(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()