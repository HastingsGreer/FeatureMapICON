import unittest
import tensorflow as tf

class TFTestCase(unittest.TestCase):
    def test_tf_gpu(self):
        self.assertTrue(tf.test.is_gpu_available())
