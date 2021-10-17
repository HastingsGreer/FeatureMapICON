import unittest
import data.DavisEval

import tensorflow.do_nothing_model

class DavisTestCase(unittest.TestCase):
    def testDoNothing(self):
        data.DavisEval(tensorflow.do_nothing_model)
        
