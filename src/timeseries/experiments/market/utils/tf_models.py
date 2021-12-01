import os
import tensorflow as tf

def save_tf_model(model, path, model_version):
    model_path = os.path.join(*path, model_version)
    tf.saved_model.save(model, model_path)
    return model_path

def load_tf_model(model_path):
    return tf.saved_model.load(model_path)