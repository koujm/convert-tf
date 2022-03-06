import tensorflow as tf
import numpy as np

from model.layers import RelativeBiases
from model.layers import SelfAttention


def is_toeplitz(matrix):
  for i in range(matrix.shape[0] - 1):
    for j in range(matrix.shape[1] - 1):
      if matrix[i, j] != matrix[i+1, j+1]:
        return False
  return True


class RelativeBiasesTest(tf.test.TestCase):

  def test_lower_max_attention(self):
    rb = RelativeBiases(max_relative_attention=2)
    inputs = np.random.random((3, 4, 4))
    output = rb(inputs)
    self.assertTrue(is_toeplitz(rb.bias))
    self.assertAllEqual(inputs.shape, output.shape)

  def test_higher_max_attention(self):
    rb = RelativeBiases(max_relative_attention=5)
    inputs = np.random.random((3, 4, 4))
    output = rb(inputs)
    self.assertTrue(is_toeplitz(rb.bias))
    self.assertAllEqual(inputs.shape, output.shape)


class SelfAttentionTest(tf.test.TestCase):
  
  def test_unk_mask(self):
    sa = SelfAttention(intermediate_dim=3, max_relative_attention=2)
    inputs = np.random.random((3, 4, 6))
    mask = np.ones((3, 4))
    mask[:,3] = 0
    output = sa(inputs, unk_mask=mask)
    self.assertAllEqual(inputs.shape, output.shape)

  def test_training(self):
    sa = SelfAttention(intermediate_dim=3, max_relative_attention=2)
    inputs = np.random.random((3, 4, 6))
    output = sa(inputs, training=True)
    self.assertAllEqual(inputs.shape, output.shape)


if __name__ == "__main__":
  tf.test.main()
