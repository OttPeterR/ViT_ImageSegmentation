import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Layer, Add, Flatten, Dropout
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model
from keras import backend as K
import numpy as np

from MakePatchesLayer import MakePatchesLayer

class ProjectPatchesLayer(Layer):
  def __init__(self, D):
    super(ProjectPatchesLayer, self).__init__()
    self.D = D
    self.dense = Dense(D, activation=None) # for linear embedding

  def call(self, patches):
    # linearly project via dense layer with no activation
    projected_patches = self.dense(patches) #shape==[batches, n_patches, D]
    return projected_patches

def test_projection():
  rng = tf.random.get_global_generator()

  test_data1 = rng.normal(shape=[5, 123, 101, 3]) # 4*4 patches
  patches = MakePatchesLayer(P=25)(test_data1) 
  projected_patches = ProjectPatchesLayer(D=128)(patches)

  p_s = patches.shape
  emb_s = projected_patches.shape
  assert p_s[0] == emb_s[0], "ERROR: batches are not correct"
  assert p_s[1] == emb_s[1], "ERROR: n_patches are not correct"
  assert emb_s[2] == 128, "ERROR: embedding dimension not correct"

  print("Projecting layer ready")