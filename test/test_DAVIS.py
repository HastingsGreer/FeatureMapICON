import unittest
import fmapicon.davis_eval
import fmapicon.utils

class DavisTestCase(unittest.TestCase):
#    def testDoNothing(self):
#        data.DavisEval(tensorflow_.fmapicon_utils.do_nothing_model, "do_nothing")
    def testFMAPICON(self):
        fmapicon.davis_eval.DavisEval(fmapicon.utils.FMAPICON_DAVIS_model("results/put_latest_model_here"), "FMAPICON")
        
