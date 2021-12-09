import unittest
import fmapicon.davis_eval
import fmapicon.utils

import urllib.request
from os.path import exists
class DavisTestCase(unittest.TestCase):
#    def testDoNothing(self):
#        data.DavisEval(tensorflow_.fmapicon_utils.do_nothing_model, "do_nothing")
    def testFMAPICON(self):
        if not exists("network_00019.trch"):
            print("Downloading pretrained model")
            urllib.request.urlretrieve(
              "https://github.com/uncbiag/FeatureMapICON/releases/download/tag/network00019.trch", "test/network00019.trch")
        fmapicon.davis_eval.DavisEval(fmapicon.utils.FmapICONSegmentationModel("test/network00019.trch"), "FMAPICON")
        
