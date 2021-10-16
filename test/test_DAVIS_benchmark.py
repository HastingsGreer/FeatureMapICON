import unittest

import data.DAVIS

class DAVISTestCase(unittest.TestCase):
    def testDAVISbenchmark(self):
        data.DAVIS.maybe_download_DAVIS()
