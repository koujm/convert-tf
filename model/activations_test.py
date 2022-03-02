import tensorflow as tf
import numpy as np

from model import activations


class ActivationsTest(tf.test.TestCase):

  def test_fast_gelu(self):

    def gelu(x):
      return x / (1 + np.exp(-1.702 * x))

    x = np.random.random((2, 5))
    result = activations.fast_gelu(tf.constant(x, dtype=tf.float32))
    expected = gelu(x)

    self.assertAllClose(result, expected, rtol=1e-05)


if __name__ == '__main__':
  tf.test.main()
