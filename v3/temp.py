import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

for var_name, _ in tf.contrib.framework.list_variables(
                './music_vae_model'):
          print(var_name)