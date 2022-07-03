import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Layer, Add, Flatten, Dropout
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model
from keras import backend as K
import numpy as np

from MakePatchesLayer import MakePatchesLayer
from ProjectPatchesLayer import ProjectPatchesLayer

class EmbedPatchesLayer(Layer):
  # linearly embedding patches into D dimensions (from P*P*C input size)
  # set n_patches so the embedding component knows how many patches there will be
  # set D as the output dimension of (embedding + linear projection of flattened patch)
  # D is the same as in the paper's equations
  def __init__(self, n_patches, embed_size):
    super(EmbedPatchesLayer, self).__init__()
    self.embed_size = embed_size
    self.n_patches = n_patches
    self.embedding = Embedding(input_dim=n_patches, output_dim=embed_size)
    self.embedding_positions = tf.constant([i for i in range(n_patches)], dtype=tf.int32)
    self.concat = layers.Concatenate(axis=2)

  def call(self, patches):
    # embedding information for each patch, this one is kinda weird...
    # it needs the same input of [1,2,3, ... n_patches] for every run but the output changes
    # because the embedding layer has learnable parameters, so self.embedding_positions is 
    # saved as a class attribute to speed up the layer
    embedding_information = tf.expand_dims(self.embedding(self.embedding_positions),0)
    n_batches = tf.shape(patches)[0]
    embedding_information = tf.repeat(embedding_information, n_batches, axis=0)
    # concat together and return
    embedded_patches = self.concat([patches, embedding_information])
    return embedded_patches


# test the embedding layer
def test_embedding():
  rng = tf.random.get_global_generator()

  D=128
  embed_size=16
  test_data1 = rng.normal(shape=[5, 123, 101, 3]) # 4*4 patches
  patches = MakePatchesLayer(P=25)(test_data1)
  projected_patches = ProjectPatchesLayer(D=D)(patches)
  embedded_patches = EmbedPatchesLayer(n_patches=16, embed_size=embed_size)(projected_patches)

  p_s = patches.shape
  emb_s = embedded_patches.shape
  assert p_s[0] == emb_s[0], "ERROR: batches are not correct"
  assert p_s[1] == emb_s[1], "ERROR: n_patches are not correct"
  assert emb_s[2] == (D+embed_size), "ERROR: embedding dimension not correct"
  print("Embedding layer ready")