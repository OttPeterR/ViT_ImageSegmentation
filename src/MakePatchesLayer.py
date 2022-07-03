import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Layer, Add, Flatten, Dropout
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model
from keras import backend as K
import numpy as np

class MakePatchesLayer(Layer):
  # cuts input image into patches
  # no learnable parameters, just set P for patch size
  # P is the same as in the papers equations
  def __init__(self, P):
    super(MakePatchesLayer, self).__init__()
    self.P = P
    
  def call(self, images):
    patches = tf.image.extract_patches(
        images=images, 
        sizes=[1, self.P, self.P, 1], # [batch, row, col, channel]
        strides=[1, self.P, self.P, 1], # [batch, row, col, channel]
        rates=[1, 1, 1, 1],
        padding="VALID", # drop patches that do not fully fit in the image
        )
    # patches.shape == [batch, patch row, patch col, (P*P*C)]
    # need to flatten out the row and cols into just one dimension 
    flattened_patches = tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])
    # flattened to [batch, n_patches, (P*P*C)]
    return flattened_patches

def test_patches():
  # test data is 5 images of 260x123 shape with 3 channels
  rng = tf.random.get_global_generator()
  test_data1 = rng.normal(shape=[5, 260, 123, 3]) # some data gets cut off
  test_data2 = rng.normal(shape=[5, 250, 100, 3]) # no data gets cut off

  # will cut into 25*25 patches
  patches_layer = MakePatchesLayer(P=25) 

  output1 = patches_layer(test_data1) 
  output2 = patches_layer(test_data2) 
  assert output1.shape == [5,40,(25*25*3)], "ERROR: patches in test 1 are not right"
  assert output2.shape == [5,40,(25*25*3)], "ERROR: patches in test 2 are not right"
  print("Patch layer ready")