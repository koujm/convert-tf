import tensorflow as tf


class RelativeBiases(tf.keras.layers.Layer):
  """Attention bias that depends only on the relative positions.

  Args:
    max_relative_attention: Maximum distance taken into account for bias.
    regularizer: Regularizer for bias.
  """

  def __init__(self,
               max_relative_attention,
               regularizer="l2"):
    super(RelativeBiases, self).__init__()
    self.max_relative_attention = max_relative_attention
    self.regularizer = regularizer

  def build(self, input_shape):
    # Relative positions can be represented by a toeplitz matrix.
    row = tf.expand_dims(tf.range(input_shape[-1]), 0)
    col = tf.expand_dims(tf.range(input_shape[-2]), 1)
    toeplitz_matrix = row - col
    toeplitz_matrix += self.max_relative_attention
    relative_positions = tf.clip_by_value(
        toeplitz_matrix, 0, self.max_relative_attention * 2)

    relative_biases = self.add_weight(
        name="relative_biases",
        shape=(self.max_relative_attention * 2 + 1,),
        initializer="zeros",
        regularizer=self.regularizer,
        trainable=True)

    self.bias = tf.identity(tf.gather(relative_biases, relative_positions))

  def call(self, inputs):
    return inputs + tf.expand_dims(self.bias, axis=0)


class SelfAttention(tf.keras.layers.Layer):
  """Self attention layer with relative position bias.

  Args:
    intermediate_dim: Size of attention head for query and key.
    max_relative_attention: Maximum distance that contributes to attention.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
  """

  def __init__(self,
               intermediate_dim,
               max_relative_attention,
               initializer="random_uniform",
               regularizer="l2"):
    super(SelfAttention, self).__init__()
    self.intermediate_dim = intermediate_dim
    self.max_relative_attention = max_relative_attention
    self.initializer = initializer
    self.regularizer = regularizer
    
    self.relative_biases = RelativeBiases(self.max_relative_attention)
  
  def build(self, input_shape):
    self.embedding_dim = input_shape[-1]
    self.max_length = input_shape[-2]

    self.query_weights = self.add_weight(
        name="query_weights",
        shape=(self.embedding_dim, self.intermediate_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)

    self.key_weights = self.add_weight(
        name="key_weights",
        shape=(self.embedding_dim, self.intermediate_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)

    self.value_weights = self.add_weight(
        name="value_weights",
        shape=(self.embedding_dim, self.embedding_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)


  def call(self, inputs, unk_mask=None, training=None):
    # Hidden layers for query, key and value.
    query = tf.tensordot(inputs, self.query_weights, axes=(2, 0))
    key = tf.tensordot(inputs, self.key_weights, axes=(2, 0))
    value = tf.tensordot(inputs, self.value_weights, axes=(2, 0))

    query_key = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.math.truediv(
        query_key, tf.math.sqrt(float(self.embedding_dim))) 

    if unk_mask is not None:
      unk_mask = tf.expand_dims(unk_mask, axis=-2)
      attention_scores = self._apply_mask(attention_scores, unk_mask)

    if not training:
      # Prevent attention from future tokens when not training.
      causal_mask = tf.ones(
          (self.max_length, self.max_length),
          dtype=inputs.dtype)
      keep = tf.math.minimum(self.max_relative_attention, self.max_length)
      causal_mask = tf.linalg.band_part(causal_mask, keep, keep)
      causal_mask = tf.expand_dims(causal_mask, axis=0)
      attention_scores = self._apply_mask(attention_scores, causal_mask)

    attention_scores = self.relative_biases(attention_scores)
    attention_scores = tf.nn.softmax(attention_scores)

    return tf.matmul(attention_scores, value)

  def _apply_mask(self, inputs, mask):
    mask = tf.cast(mask, inputs.dtype)
    mask = 1 - mask
    if inputs.dtype is tf.float16:
      inputs -= 65504. * mask 
    else:
      inputs -= 1.e9 * mask 
    return inputs
