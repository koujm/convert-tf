import tensorflow as tf


def fast_gelu(x):
  """Fast Gaussian error linear unit (GELU) activation function.

  Args:
    x: Input tensor.

  Returns:
    The gaussian error linear activation tensor.

  Reference:
    - [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
  """
  return x * tf.math.sigmoid(1.702 * x)


def get(identifier):
  if "fast_gelu" == identifier:
    return fast_gelu
  
  return tf.keras.activations.get(identifier)
