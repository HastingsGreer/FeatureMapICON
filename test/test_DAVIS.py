import unittest
import data

import tensorflow_.fmapicon_utils

class DavisTestCase(unittest.TestCase):
    def testDoNothing(self):
        data.DavisEval(tensorflow_.fmapicon_utils.do_nothing_model, "do_nothing")
        
