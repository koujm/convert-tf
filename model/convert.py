from collections import namedtuple

import tensorflow as tf
import tensorflow_text as tf_text

from model.layers import MultiHeadAttention
from model.layers import TransformerBlock
from model.layers import PositionalEncodings
from model.layers import HiddenLayer
from model.layers import FeatureEncoder


HParams = namedtuple("HParams", (
  "vocab_path",
  "context_feature",
  "response_feature",
  "max_sequence_length",
  "embedding_size",
  "max_attention_spans",
  "transformer_dims",
  "final_dim",
  )
)


UNK_ID = 0


class ConveRT(tf.keras.Model):
  
  def __init__(self, hparams, **kwargs):
    super(ConveRT, self).__init__(**kwargs)
    self.hparams = hparams

    with tf.io.gfile.GFile(self.hparams.vocab_path) as f:
      vocab = f.read().splitlines()
    self.tokenizer = tf_text.FastWordpieceTokenizer(
        vocab=vocab,
        suffix_indicator="##",
        token_out_type=tf.int32,
        support_detokenization=False)

    self.regularizer = tf.keras.regularizers.L2(1e-5)

    self.embedding_layer = tf.keras.layers.Embedding(
      input_dim=len(vocab),
      output_dim=self.hparams.embedding_size,
      embeddings_regularizer=self.regularizer)

    self.positional_encoding = PositionalEncodings(
        positional_dims=(47, 11),
        regularizer=self.regularizer)
    
    self.layer_norm_input = tf.keras.layers.LayerNormalization()

    self.transformer_blocks = []
    for attention_span in self.hparams.max_attention_spans:
      self.transformer_blocks.append(
          TransformerBlock(
            intermediate_dim=self.hparams.transformer_dims[0],
            max_relative_attention=attention_span,
            hidden_dim=self.hparams.transformer_dims[1],
            regularizer=self.regularizer)
          )

    self.multihead_attention = MultiHeadAttention(
      num_heads=2,
      intermediate_dim=self.hparams.transformer_dims[0],
      max_relative_attention=self.hparams.max_attention_spans[-1],
      regularizer=self.regularizer)

    self.context_feat_encoder = FeatureEncoder(
      num_hiddens=3,
      output_dim=self.hparams.final_dim,
      regularizer=self.regularizer)

    self.response_feat_encoder = FeatureEncoder(
      num_hiddens=3,
      output_dim=self.hparams.final_dim,
      regularizer=self.regularizer)

  def call(self, inputs, training=False):
    with tf.name_scope("embed_context"):
      context_embedding = self._embed_nl(
        inputs[self.hparams.context_feature], training)

    with tf.name_scope("embed_response"):
      response_embedding = self._embed_nl(
        inputs[self.hparams.response_feature], training)

    encoded_context = self.context_feat_encoder(context_embedding)
    encoded_response = self.response_feat_encoder(response_embedding)

    return tf.matmul(encoded_context, encoded_response, transpose_b=True)

  def _embed_nl(self, inputs, training):
    with tf.name_scope("embed_nl"):
      x = tf.strings.lower(inputs)
      x = tf.strings.regex_replace(x, r"\d{5,}", "#")
      x = self.tokenizer.tokenize(x)
      x = x.to_tensor(default_value=UNK_ID)

      if training:
        x = x[:,:tf.math.minimum(x.shape[1]), self.hparams.max_sequence_length]

      unk_mask = tf.math.not_equal(x, UNK_ID)      

      x = self.embedding_layer(x)
      x *= tf.math.sqrt(float(self.hparams.embedding_size))

      x = self.positional_encoding(x)
      x = self.layer_norm_input(x)

      for transformer_block in self.transformer_blocks:
        x = transformer_block(x, unk_mask, training)

      x = self.multihead_attention(x, unk_mask, training)

      return x
