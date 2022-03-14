import tensorflow as tf
import numpy as np

from model.layers import RelativeBiases
from model.layers import SelfAttention
from model.layers import MultiHeadAttention
from model.layers import TransformerBlock


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
    self.assertAllEqual(inputs.shape, output.shape)

  def test_higher_max_attention(self):
    rb = RelativeBiases(max_relative_attention=5)
    inputs = np.random.random((3, 4, 4))
    output = rb(inputs)
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
    mask = np.ones((3, 4))
    mask[:,3] = 0
    output = sa(inputs, unk_mask=mask, training=True)
    self.assertAllEqual(inputs.shape, output.shape)


class MultiHeadAttentionTest(tf.test.TestCase):

  def test_output_shape(self):
    num_heads = 2
    mha = MultiHeadAttention(
        num_heads=num_heads,
        intermediate_dim=3,
        max_relative_attention=2)
    inputs = np.random.random((3, 4, 6))
    mask = np.ones((3, 4))
    mask[:,3] = 0
    output = mha(inputs, unk_mask=mask, training=True)
    self.assertAllEqual((3, 6 * num_heads), output.shape)


class TransformerBlockTest(tf.test.TestCase):

  def test_output_shape(self):
    tb = TransformerBlock(
        intermediate_dim=3,
        max_relative_attention=2,
        hidden_dim=9)
    inputs = np.random.random((3, 4, 6))
    mask = np.ones((3, 4))
    mask[:,3] = 0
    output = tb(inputs, unk_mask=mask, training=True)
    self.assertAllEqual(inputs.shape, output.shape)


if __name__ == "__main__":
  tf.test.main()
