# convert-tf
ConveRT (Conversational Representations from Transformers) TF model based on [https://arxiv.org/pdf/1911.03688.pdf](https://arxiv.org/pdf/1911.03688.pdf).

## Usage

### Training

Below is the code to train a new model with the same default settings as described in the paper.

```python
from model import convert

model = convert.get_compiled_model(vocab_path, max_steps)
model.fit(inputs)
```

where:
  * vocab_path (str): path to any WordPiece compatible vocab file with "##" suffix indicator, or build one by running [tools/build_vocab.py](tools/build_vocab.py) on the train input corpus.
  * max_steps (int): number of training steps before stopping learning rate decay.
  * inputs: a dictionary mapping `"context"` and `"response"` keys to the corresponding input string array/tensors or a `tf.data.Dataset` of the same format (check Keras [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) API).

Input specs:
```python
{
  "context": tf.TensorSpec(shape=(None,), dtype=tf.string),
  "response": tf.TensorSpec(shape=(None,), dtype=tf.string)
}
```

A sample training code can be found in [tools/test_train.py](tools/test_train.py).

### Inference

An `ExportSavedModel` callback is provided in [model/utils.py](model/utils.py) to export the model for inference.

The saved model can then be loaded in the following ways for inference:

```python
import tensorflow as tf
import tensorflow_text as tf_text
from model import convert

inputs = {"context": ..., "response": ...}

# Option 1
model = tf.saved_model.load(saved_model)
serve_fn = model.signatures["serve"]
output = serve_fn(context=inputs["context"], response=inputs["response"])

# Option 2
model = tf.keras.models.load_model(
  saved_model,
  custom_objects={"ConveRT": convert.ConveRT}
)
output = model(inputs)
```

## Note

1. Model supports single context currently.
2. Only 1 OOV bucket.
