import os

import tensorflow as tf

from model.convert import HParams
from model.convert import ConveRT


class ConvertTest(tf.test.TestCase):

  def test_build(self):
    test_dir = os.path.join(
        os.path.dirname(__file__),
        "../test_data")

    hparams = HParams(
        vocab_path=os.path.join(test_dir, "vocab.txt"),
        max_sequence_length=10,
        embedding_size=8,
        max_attention_spans=[2, 3, 4, 4],
        transformer_dims=[4, 16],
        final_dim=8,
        dropout_rate=None,
        )
    model = ConveRT(hparams)

    with open(os.path.join(test_dir, "random.txt")) as f:
      random_text = f.read().splitlines()

    inputs = {
        "context": random_text[:2],
        "response": random_text[-2:],
        }

    output = model(inputs, training=True)

    self.assertAllEqual((2, 2), output["prediction"].shape)


if __name__ == "__main__":
  tf.test.main()
