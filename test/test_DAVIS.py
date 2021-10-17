import unittest
import data

import tensorflow.fmapicon_utils

class DavisTestCase(unittest.TestCase):
    def testDoNothing(self):
        data.DavisEval(tensorflow.fmapicon_utils.do_nothing_model)
        
