import tensorflow as tf


class ExportSavedModel(tf.keras.callbacks.Callback):

  def __init__(self, output_dir):
    super(ExportSavedModel, self).__init__()
    self.output_dir = output_dir

  def on_test_end(self, logs=None):
    arg_specs, kwarg_specs = self.model.save_spec()

    self.model.save(
        self.output_dir,
        overwrite=False,
        include_optimizer=False,
        save_format="tf",
        save_traces=False,
        signatures={
          "serve": self.model.serve.get_concrete_function(
            *arg_specs, **kwarg_specs),
          },
        )
