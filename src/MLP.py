import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Layer, Add, Flatten, Dropout
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model
from keras import backend as K
import numpy as np


# Multi Layer Perceptron
# "MLP" in the paper
class MLP(Layer):
  def __init__(self, units=[128,128], dropout=0.0, activation="gelu"):
    super(MLP, self).__init__()
    self.layers = []
    assert len(units) > 0, "MLP: need more layers"
    for u in units:
      self.layers += [Dense(u, activation=activation)]
      self.layers += [Dropout(dropout)]

  def call(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

def test_mlp():
  my_mlp = MLP(units=[10,20,30], dropout=0.1)
  rng = tf.random.get_global_generator()
  test_data1 = rng.normal(shape=[5, 8, 1024])
  assert my_mlp(test_data1).shape[0] == 5,  "ERROR: batch dimension not preserved"
  assert my_mlp(test_data1).shape[1] == 8,  "ERROR: channel dimension not preserved"
  assert my_mlp(test_data1).shape[2] == 30, "ERROR: output dimension incorrect"
  print("MLP is ready")