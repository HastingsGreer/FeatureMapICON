import unittest
import data

import tensorflow_.fmapicon_utils

class DavisTestCase(unittest.TestCase):
#    def testDoNothing(self):
#        data.DavisEval(tensorflow_.fmapicon_utils.do_nothing_model, "do_nothing")
    def testFMAPICON(self):
        data.DavisEval(tensorflow_.fmapicon_utils.FMAPICON_model("../tensorflow_/results/log_probability_2/"), "FMAPICON")
        
