def build_ViT_ImageSeg(image_edge, P, D, D2, spatial_emb, features_per_patch, n_transformer_blocks, n_attention_heads, MLP_units, final_encoder_MLP_units, decoder_MLP_units):
  n_patches=(image_edge//P)**2
  MLP_units += [D+spatial_emb]
  final_encoder_MLP_units += [features_per_patch*n_patches]
  epsilon=1e-6

  ##################################################################
  # IMAGE INGEST AND EMBEDDING
  input_image = keras.Input(shape=(image_edge, image_edge, 3))
  patches = MakePatchesLayer(P=P)(input_image)
  projected_patches = ProjectPatchesLayer(D=D)(patches)
  embedded_patches = EmbedPatchesLayer(n_patches=n_patches, embed_size=spatial_emb)(projected_patches)

  ##################################################################
  # ViT ENCODER
  x = embedded_patches
  for _ in range(n_transformer_blocks):
    x1 = LayerNormalization(epsilon=epsilon)(x)
    attn = MultiHeadAttention(num_heads=n_attention_heads, key_dim=D, dropout=0.2)(x1,x1)
    msa_block_output = attn + x # skip connection

    norm_attn = LayerNormalization(epsilon=epsilon)(msa_block_output)
    mlp_features = MLP(MLP_units, dropout=0.4)(norm_attn)
    x = mlp_features + msa_block_output # another skip connection

  encoder_features = Sequential([
        LayerNormalization(epsilon=epsilon),
        Flatten(),
        Dropout(0.2),
        MLP(final_encoder_MLP_units, dropout=0.4),
      ])(x)
  patch_features = layers.Reshape((n_patches, features_per_patch))(encoder_features)


  ##################################################################
  # DECODER
  # def repeat_across_batch(features, inputs):
  #   # batch_size = tf.shape(inputs)[0]
  #   features = tf.expand_dims(features, 1)
  #   features = tf.repeat(features, n_patches, axis=1)
  #   return features
  # patch_encodings = layers.Lambda(lambda x: repeat_across_batch(x[0], [1]))([encoder_features,input_image])
  # bring the patches from the input image
  # this is for fidelity, like U-Net does
  projected_patches2 = ProjectPatchesLayer(D=D2)(patches)
  embeded_patches2 = EmbedPatchesLayer(n_patches=n_patches, embed_size=spatial_emb)(projected_patches2)
  concat_patches = layers.Concatenate(axis=2)([embeded_patches2, patch_features])
  # shape == (n_patches, concat_features)
  output_image_patches = Sequential([
        LayerNormalization(epsilon=epsilon),
        MLP(decoder_MLP_units, dropout=0.3, activation="relu"),
        Dense(P*P*3, activation="hard_sigmoid") # project patch features 0..1
      ])(concat_patches)
  output_image = layers.Reshape((n_patches, (P*P), 3))(output_image_patches)
  output_image = layers.Reshape((image_edge, image_edge, 3))(output_image)
  return Model(input_image, output_image)

def test_vit():
  image_edge=128
  test_model = build_ViT_ImageSeg(
      # patching
      image_edge=image_edge, # edge of input/output image
      spatial_emb=8,
      P=16, # pixel edge per patch
      D=16, # patch projection latent size
      
      # transformer 
      n_transformer_blocks=2,
      n_attention_heads=4,
      MLP_units=[64,128], # encoder MLP
      final_encoder_MLP_units=[128,128],
      features_per_patch=32,

      # decoder
      D2=32, # decoder patch projection latent size
      decoder_MLP_units=[128,128],
  )

  rng = tf.random.get_global_generator()
  test_data = rng.normal(shape=[7, image_edge, image_edge, 3])
  outputs = test_model(test_data)
  assert test_data.shape[1]==outputs.shape[1], "image size was changed"
  assert test_data.shape[2]==outputs.shape[2], "image size was changed"
  print("ViT for Image Seg is ready")
