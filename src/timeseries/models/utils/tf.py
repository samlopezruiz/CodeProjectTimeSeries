import os
import tensorflow as tf


def plot_tf_model(model, folder, image_name='model', show_shapes=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    tf.keras.utils.plot_model(
        model, to_file=os.path.join(folder, image_name+'.png'), show_shapes=show_shapes, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )