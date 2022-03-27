import argparse
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from model import convert
from model import utils


parser = argparse.ArgumentParser()
parser.add_argument("--vocab_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)


def show_summary(model):
  inputs = {
      "context": tf.keras.Input(
        type_spec=tf.TensorSpec(shape=(None,), dtype=tf.string)
        ),
      "response": tf.keras.Input(
        type_spec=tf.TensorSpec(shape=(None,), dtype=tf.string)
        ),
      }
  model(inputs)
  model.summary()


def get_dataset():
  train_ds = tfds.load("glue/qnli", split="train")
  train_ds = train_ds.filter(lambda x: x["label"] == 0)
  train_ds = train_ds.map(
      lambda x: {"context": x["question"], "response": x["sentence"]}
      )
  train_ds = train_ds.shuffle(1000).batch(100, drop_remainder=True)

  val_ds = tfds.load("glue/qnli", split="validation")
  val_ds = val_ds.filter(lambda x: x["label"] == 0)
  val_ds = val_ds.map(
      lambda x: {"context": x["question"], "response": x["sentence"]}
      )
  val_ds = val_ds.shuffle(1000).batch(100, drop_remainder=True)

  return train_ds, val_ds


def train(model, train_dataset, val_dataset, output_dir):
  callbacks = [
      tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(output_dir, "log"),
        write_graph=True,
        write_steps_per_second=True,
        update_freq="epoch"),
      tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, "ckpt"),
        save_best_only=True,
        save_freq="epoch"),
      utils.ExportSavedModel(
        output_dir=os.path.join(output_dir, "saved_model")),
      ]

  history = model.fit(
      train_dataset,
      epochs=1,
      callbacks=callbacks,
      validation_data=val_dataset)


if __name__ == "__main__":
  args = parser.parse_args()

  model = convert.get_compiled_model(
      args.vocab_path,
      max_steps=10 * 100)

  show_summary(model)

  train_ds, val_ds = get_dataset()
  train(model, train_ds, val_ds, args.output_dir)
