"""Build a subword vocabulary for WordPiece tokenizers.

TODO: add option to pretokenize text for languages where no explicit spaces
exist between words.
"""
import argparse
import collections
import re

import tensorflow as tf

from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner


parser = argparse.ArgumentParser()
parser.add_argument("--input_filepattern", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--max_vocab_size", type=int, required=True)
parser.add_argument("--num_iterations", type=int, default=4)
parser.add_argument("--min_subword_count", type=int, default=250)
parser.add_argument("--max_subword_length", type=int, default=20)


def encode(text):
  """Encode a unicode string as a list of tokens.

  Args:
    text: a unicode string
    
  Returns:
    a list of tokens as Unicode strings
  """
  if not text:
    return []

  # Limit more than 4 consecutive digits. 
  text = re.sub(r"\d{5,}", "#", text.lower().strip())

  ret = []
  token_start = 0
  for pos in range(1, len(text)):
    if text[pos].isalnum() != text[pos - 1].isalnum() or (
        text[pos].strip() == ""):
      token = text[token_start:pos]
      if token.strip() != "":
        ret.append(token.strip())
      token_start = pos

  if token_start < len(text):
    final_token = text[token_start:]
    ret.append(final_token.strip())

  return ret


def _get_text(filenames):
  for filename in filenames:
    with tf.io.gfile.GFile(filename) as f:
      while line := f.readline():
        yield line.strip()


def _save_vocab(vocab, path):
  with tf.io.gfile.GFile(path, "w") as f:
    for v in vocab:
      f.write(v + "\n")


if __name__ == "__main__":
  args = parser.parse_args()
  filenames = tf.io.gfile.glob(args.input_filepattern)
  token_counts = collections.Counter()
  for line in _get_text(filenames):
    token_counts.update(encode(line))
  vocab = learner.learn(
      token_counts,
      args.max_vocab_size,
      reserved_tokens=["[UNK]"],
      lower_thresh=args.min_subword_count,
      num_iterations=args.num_iterations,
      max_token_length=args.max_subword_length,
      max_unique_chars=10000
      )
  _save_vocab(vocab, args.output_path)
