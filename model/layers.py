import tensorflow as tf

from model import activations


class RelativeBiases(tf.keras.layers.Layer):
  """Attention bias that depends only on the relative positions.

  Args:
    max_relative_attention: Maximum distance taken into account for bias.
    regularizer: Regularizer for bias.
  """

  def __init__(self,
               max_relative_attention,
               regularizer="l2",
               **kwargs):
    super(RelativeBiases, self).__init__(**kwargs)
    self.max_relative_attention = max_relative_attention
    self.regularizer = regularizer

    self.relative_biases = self.add_weight(
        name="relative_biases",
        shape=(self.max_relative_attention * 2 + 1,),
        initializer="zeros",
        regularizer=self.regularizer,
        trainable=True)

  def call(self, inputs):
    # Relative positions can be represented by a toeplitz matrix.
    row = tf.expand_dims(tf.range(inputs.shape[-1]), axis=0)
    col = tf.expand_dims(tf.range(inputs.shape[-2]), axis=1)
    toeplitz_matrix = row - col
    toeplitz_matrix += self.max_relative_attention

    relative_positions = tf.clip_by_value(
        toeplitz_matrix, 0, self.max_relative_attention * 2)
    bias = tf.identity(tf.gather(self.relative_biases, relative_positions))

    return inputs + tf.expand_dims(bias, axis=0)


class SelfAttention(tf.keras.layers.Layer):
  """Self attention layer with relative position bias.

  Args:
    intermediate_dim: Size of attention head for query and key.
    max_relative_attention: Maximum distance that contributes to attention.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
    multi: Whether output is used for multi-headed attention.
  """

  def __init__(self,
               intermediate_dim,
               max_relative_attention,
               initializer="random_uniform",
               regularizer="l2",
               multi=None,
               **kwargs):
    super(SelfAttention, self).__init__(**kwargs)
    self.intermediate_dim = intermediate_dim
    self.max_relative_attention = max_relative_attention
    self.initializer = initializer
    self.regularizer = regularizer
    self.multi = multi
    
    self.relative_biases = RelativeBiases(self.max_relative_attention)
  
  def build(self, input_shape):
    self.embedding_dim = input_shape[-1]

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

    if not self.multi:
      self.value_weights = self.add_weight(
          name="value_weights",
          shape=(self.embedding_dim, self.embedding_dim),
          initializer=self.initializer,
          regularizer=self.regularizer,
          trainable=True)


  def call(self, inputs, unk_mask, training=None):
    # Hidden layers for query, key and value.
    query = tf.tensordot(inputs, self.query_weights, axes=(2, 0))
    key = tf.tensordot(inputs, self.key_weights, axes=(2, 0))

    query_key = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.math.truediv(
        query_key, tf.math.sqrt(float(self.embedding_dim))) 

    unk_mask = tf.cast(unk_mask, inputs.dtype)
    attention_scores = self._apply_mask(
        attention_scores,
        tf.expand_dims(unk_mask, axis=-2)
        )

    if not training:
      # Prevent attention from future tokens when not training.
      max_length = inputs.shape[-2]
      causal_mask = tf.ones(
          (max_length, max_length),
          dtype=inputs.dtype)
      keep = tf.math.minimum(self.max_relative_attention, max_length)
      causal_mask = tf.linalg.band_part(causal_mask, keep, keep)
      attention_scores = self._apply_mask(
          attention_scores,
          tf.expand_dims(causal_mask, axis=0)
          )

    attention_scores = self.relative_biases(attention_scores)
    attention_scores = tf.nn.softmax(attention_scores)

    if not self.multi:
      value = tf.tensordot(inputs, self.value_weights, axes=(2, 0))
      return tf.matmul(attention_scores, value)

    return tf.squeeze(
        tf.matmul(self._reduce_sqrtn(attention_scores, unk_mask), inputs)
        )

  def _apply_mask(self, inputs, mask):
    mask = 1 - mask

    if inputs.dtype is tf.float16:
      inputs -= 65504. * mask 
    else:
      inputs -= 1.e9 * mask

    return inputs

  def _reduce_sqrtn(self, scores, weights):
    scores *= tf.expand_dims(weights, axis=-1) 
    scores = tf.math.reduce_sum(scores, axis=-2, keepdims=True)

    weights = tf.math.reduce_sum(weights, axis=-1, keepdims=True)
    weights = tf.math.maximum(weights, 1)
    weights = tf.math.rsqrt(weights)
    weights = tf.expand_dims(weights, axis=-1)

    return scores * weights
    

class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-head attention layer that concatenates sub attention layer outputs.

  Args:
    num_heads: Number of attention heads.
    intermediate_dim: Size of attention head for query and key.
    max_relative_attention: Maximum distance that contributes to attention.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
  """

  def __init__(self,
               num_heads,
               intermediate_dim,
               max_relative_attention,
               initializer="random_uniform",
               regularizer="l2",
               **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)

    if num_heads <= 0:
      raise ValueError("num_heads must be > 0")

    self.attentions = []
    for i in range(num_heads):
      self.attentions.append(
          SelfAttention(
            intermediate_dim=intermediate_dim,
            max_relative_attention=max_relative_attention,
            initializer=initializer,
            regularizer=regularizer,
            multi=True)
          )

  def call(self, inputs, unk_mask, training=None):
    outputs = []
    for attention in self.attentions:
      outputs.append(
          attention(inputs, unk_mask, training=training)
          )

    return tf.concat(outputs, axis=-1)


class TransformerBlock(tf.keras.layers.Layer):
  """Transformer block.

  Args:
    intermediate_dim: Size of attention head for query and key.
    max_relative_attention: Maximum distance that contributes to attention.
    hidden_dim: Hidden layer dimension after self attention.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
    activation: Activation function name to be used.
  """

  def __init__(self,
               intermediate_dim,
               max_relative_attention,
               hidden_dim,
               initializer="random_uniform",
               regularizer="l2",
               activation="fast_gelu",
               **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.attention = SelfAttention(
        intermediate_dim=intermediate_dim,
        max_relative_attention=max_relative_attention,
        initializer=initializer,
        regularizer=regularizer)
    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    self.activation_fn = activations.get(activation)
    
    self.initializer = initializer
    self.regularizer = regularizer
    self.hidden_dim = hidden_dim

  def build(self, input_shape):
    self.embedding_dim = input_shape[-1]

    self.weights_1 = self.add_weight(
        name="weights_1",
        shape=(self.embedding_dim, self.hidden_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)

    self.weights_2 = self.add_weight(
        name="weights_2",
        shape=(self.hidden_dim, self.embedding_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)

  def call(self, inputs, unk_mask, training=None):
    output = self.attention(inputs, unk_mask, training=training)
    output += inputs
    output = self.layer_norm_1(output)
    output = tf.tensordot(output, self.weights_1, axes=(2, 0))
    output = self.activation_fn(output)
    output = tf.tensordot(output, self.weights_2, axes=(2, 0))
    output += inputs
    output = self.layer_norm_2(output)
    return output


class PositionalEncodings(tf.keras.layers.Layer):
  """Add positional encodings to input.

  Args:
    positional_dims: List of dimensions of positional encodings.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
  """

  def __init__(self,
               positional_dims,
               initializer="random_uniform",
               regularizer="l2",
               **kwargs):
    super(PositionalEncodings, self).__init__(**kwargs)
    self.positional_dims = positional_dims
    self.initializer = initializer
    self.regularizer = regularizer

  def build(self, input_shape):
    self.embedding_dim = input_shape[-1]

    self.positional_encodings = []
    for i, positional_dim in enumerate(self.positional_dims):
      self.positional_encodings.append(
          self.add_weight(
            name=f"positional_encoding_{i}",
            shape=(positional_dim, self.embedding_dim),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True)
          )

  def call(self, inputs):
    max_length = inputs.shape[-2]

    projected_encodings = []
    for positional_encoding in self.positional_encodings:
      repeat = max_length // positional_encoding.shape[0] + 1
      projected_encodings.append(
          tf.tile(positional_encoding, multiples=(repeat, 1))[:max_length]
          )

    projected_encodings = tf.math.add_n(projected_encodings)
    return inputs + tf.expand_dims(projected_encodings, axis=0)


class HiddenLayer(tf.keras.layers.Layer):
  """Hidden layer that also supports residual connnection and layer norm.

  Args:
    hidden_dim: Hidden layer dimension.
    activation: Activation function name to be used.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
    residual: Whether to add residual connection.
    layer_norm: Whether to add layer normalization.
  """
  def __init__(self,
               hidden_dim=None,
               activation="fast_gelu",
               initializer="orthogonal",
               regularizer="l2",
               residual=True,
               layer_norm=True,
               **kwargs):
    super(HiddenLayer, self).__init__(**kwargs)
    self.hidden_dim = hidden_dim
    self.initializer = initializer
    self.regularizer = regularizer
    self.residual = residual
    self.layer_norm = (
        tf.keras.layers.LayerNormalization()
        if layer_norm else None)
    self.activation_fn = activations.get(activation)

  def build(self, input_shape):
    if self.residual:
      if self.hidden_dim is None:
        self.hidden_dim = input_shape[-1]

      if self.hidden_dim != input_shape[-1]:
        raise ValueError("hidden_dim should match input's last dimension "
                         "for residual connection.")

    self.w = self.add_weight(
        name="weights",
        shape=(input_shape[-1], self.hidden_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)

  def call(self, inputs):
    output = tf.matmul(inputs, self.w)
    output = self.activation_fn(output)

    if self.residual:
      output += inputs

    if self.layer_norm:
      output = self.layer_norm(output)

    return output


class FeatureEncoder(tf.keras.layers.Layer):
  """Encode feature with hidden layers and skip connection.

  Args:
    num_hiddens: Number of hidden layers.
    output_dim: Dimension of output's last axis.
    initializer: Initializer for weights.
    regularizer: Regularizer for weights.
  """
  def __init__(self,
               num_hiddens,
               output_dim,
               initializer="orthogonal",
               regularizer="l2",
               **kwargs):
    super(FeatureEncoder, self).__init__(**kwargs)
    self.output_dim = output_dim
    self.initializer = initializer
    self.regularizer = regularizer

    self.layer_norm_input = tf.keras.layers.LayerNormalization()
    self.layer_norm_output = tf.keras.layers.LayerNormalization()

    self.layers = []
    for i in range(num_hiddens):
      self.layers.append(
          HiddenLayer(
            initializer=initializer,
            regularizer=regularizer)
          )

    self.layers.append(
        HiddenLayer(
            self.output_dim,
            activation=None,
            initializer=initializer,
            regularizer=regularizer,
            residual=False,
            layer_norm=False)
        )

  def build(self, input_shape):
    self.skip_weights = self.add_weight(
        name="weights",
        shape=(input_shape[-1], self.output_dim),
        initializer=self.initializer,
        regularizer=self.regularizer,
        trainable=True)

  def call(self, inputs):
    output = self.layer_norm_input(inputs)

    for layer in self.layers:
      output = layer(output)

    skip_connection = tf.matmul(inputs, self.skip_weights)
    output += skip_connection

    output = self.layer_norm_output(output)
    output = tf.math.l2_normalize(output, axis=-1)

    return output
