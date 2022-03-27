from collections import namedtuple

import tensorflow as tf
import tensorflow_text as tf_text

from model.layers import MultiHeadAttention
from model.layers import TransformerBlock
from model.layers import PositionalEncodings
from model.layers import FeatureEncoder


HParams = namedtuple("HParams", (
  "vocab_path",
  "max_sequence_length",
  "embedding_size",
  "max_attention_spans",
  "transformer_dims",
  "final_dim",
  "dropout_rate",
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
      embeddings_regularizer=self.regularizer,
      name="SubwordEmbedding")

    self.positional_encoding = PositionalEncodings(
        positional_dims=(47, 11),
        regularizer=self.regularizer)
    
    self.layer_norm_input = tf.keras.layers.LayerNormalization(
        name="layer_norm_input")

    self.transformer_blocks = []
    for i, attention_span in enumerate(self.hparams.max_attention_spans):
      self.transformer_blocks.append(
          TransformerBlock(
            intermediate_dim=self.hparams.transformer_dims[0],
            max_relative_attention=attention_span,
            hidden_dim=self.hparams.transformer_dims[1],
            regularizer=self.regularizer,
            dropout=self.hparams.dropout_rate,
            name=f"transformer_block_{i}")
          )

    self.multihead_attention = MultiHeadAttention(
      num_heads=2,
      intermediate_dim=self.hparams.transformer_dims[0],
      max_relative_attention=self.hparams.max_attention_spans[-1],
      regularizer=self.regularizer,
      dropout=self.hparams.dropout_rate,
      name="multi_head_attention")

    self.context_feat_encoder = FeatureEncoder(
      num_hiddens=3,
      output_dim=self.hparams.final_dim,
      regularizer=self.regularizer,
      name="encode_context")

    self.response_feat_encoder = FeatureEncoder(
      num_hiddens=3,
      output_dim=self.hparams.final_dim,
      regularizer=self.regularizer,
      name="encode_response")

  def call(self, inputs, training=False):
    with tf.name_scope("embed_context"):
      context_embedding, seq_encoding = self._embed_nl(
          inputs["context"], training)

    with tf.name_scope("embed_response"):
      response_embedding, _ = self._embed_nl(
          inputs["response"], training)

    encoded_context = self.context_feat_encoder(context_embedding)
    encoded_response = self.response_feat_encoder(response_embedding)

    pred = tf.matmul(
        encoded_context,
        encoded_response,
        transpose_b=True,
        name="prediction")

    output = {"prediction": pred}

    if not training:
      output.update({
        "sequence_encoding": tf.identity(
          seq_encoding, name="sequence_encoding"),
        "text_encoding": tf.identity(
          tf.math.l2_normalize(
            context_embedding, axis=-1, name="normalize_text"),
          name="text_encoding"),
        "context_encoding": tf.identity(
          encoded_context, name="context_encoding"),
        "response_encoding": tf.identity(
          encoded_response, name="response_encoding"),
        })

    return output

  def _embed_nl(self, inputs, training):
    with tf.name_scope("embed_nl"):
      x = tf.strings.lower(inputs)
      x = tf.strings.regex_replace(x, r"\d{5,}", "#")
      x = self.tokenizer.tokenize(x)
      x = x.to_tensor(default_value=UNK_ID)

      if training:
        x = x[
            :,
            :tf.math.minimum(
              tf.shape(x)[1],
              self.hparams.max_sequence_length)
            ]

      unk_mask = tf.math.not_equal(x, UNK_ID)      

      x = self.embedding_layer(x)
      x *= tf.math.sqrt(float(self.hparams.embedding_size))

      x = self.positional_encoding(x)
      x = self.layer_norm_input(x)

      for transformer_block in self.transformer_blocks:
        x = transformer_block(x, unk_mask, training)

      seq_encoding = x

      x = self.multihead_attention(x, unk_mask, training)

      return x, seq_encoding

  def train_step(self, data):
    x = data

    with tf.GradientTape() as tape:
      output = self(x, training=True)
      y_pred = output["prediction"]
      y = tf.eye(y_pred.shape[0], name="label")

      with tf.name_scope("loss"):
        loss = self.compute_loss(x, y, y_pred)

    with tf.name_scope("gradients"):
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)

      grads_and_vars = [
          (tf.clip_by_value(grad, -1, 1), var)
          if "SubwordEmbedding" in var.name
          else (grad, var)
          for grad, var in zip(gradients, trainable_vars)
          ]
      self.optimizer.apply_gradients(grads_and_vars)

    with tf.name_scope("metrics"):
      metrics = self.compute_metrics(x, y, y_pred, sample_weight=None)

    return metrics

  def test_step(self, data):
    x = data

    output = self(x, training=False)
    y_pred = output["prediction"]
    y = tf.eye(y_pred.shape[0], name="label")

    with tf.name_scope("loss"):
      self.compute_loss(x, y, y_pred)

    with tf.name_scope("metrics"):
      metrics = self.compute_metrics(x, y, y_pred, sample_weight=None)

    return metrics

  def predict_step(self, data):
    x = data

    output = self(x, training=False)
    y_pred = output["prediction"]

    return y_pred

  def get_config(self):
    return {"hparams": self.hparams._asdict()}

  @classmethod
  def from_config(cls, config):
    return cls(HParams(**config["hparams"]))

  @tf.function
  def serve(self, *args, **kwargs):
    output = self(*args, **kwargs)
    return output


def get_compiled_model(vocab_path,
                       max_steps,
                       steps_per_execution=1):
  optimizer = tf.keras.optimizers.Adadelta(
    learning_rate=tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=1., decay_steps=max_steps, alpha=0.001),
    rho=0.9,
    )

  loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.2,
  )

  hparams = HParams(
      vocab_path=vocab_path,
      max_sequence_length=60,
      embedding_size=512,
      max_attention_spans=[3, 5, 48, 48, 48, 48],
      transformer_dims=[64, 2048],
      final_dim=512,
      dropout_rate=None,
      )

  model = ConveRT(hparams, name="ConveRT")
  model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=tf.keras.metrics.CategoricalAccuracy(),
      run_eagerly=False,
      steps_per_execution=steps_per_execution,
      )

  return model
