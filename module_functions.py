def cnn_entities(ds_encoder,ds_decoder,last_layer_decoder_activation,ds_nn,last_layer_nn_activation,samples_shape):
  encoder = []
  for i in range(0,len(ds_encoder)):
    if i == 0:
      input_shape = samples_shape[1:]
      encoder.append(tf.keras.layers.Conv2D(filters=ds_encoder[i], input_shape = input_shape, kernel_size=(3,3), padding='same'))
    else:
      encoder.append(tf.keras.layers.Conv2D(filters=ds_encoder[i], kernel_size=(3,3), padding='same'))
    
    encoder.append(default_activation)
    encoder.append(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
  encoder = tf.keras.models.Sequential(encoder)

  decoder = []
  for i in range(0,len(ds_decoder)):
    if i == 0:
      if len(encoder.layers) > 0:
        input_shape = encoder.output_shape[1:]
      else:
        input_shape = samples_shape[1:]
      decoder.append(tf.keras.layers.Conv2D(filters=ds_decoder[i], input_shape = input_shape, kernel_size=(3,3), padding='same'))
    else:
      decoder.append(tf.keras.layers.Conv2D(filters=ds_decoder[i], kernel_size=(3,3), padding='same'))
    # if i == len(ds_decoder) - 1:
    #   decoder.append(last_layer_decoder_activation)
    # else:
    decoder.append(default_activation)
    decoder.append(tf.keras.layers.UpSampling2D(size=(2,2)))
  if len(ds_decoder) > 0:
    decoder.append(tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same'))
    decoder.append(last_layer_decoder_activation)
  decoder = tf.keras.models.Sequential(decoder)

  if len(encoder.layers) == 0:
    autoencoder = tf.keras.models.Sequential([])
  else:
    autoencoder = tf.keras.models.Sequential([encoder, decoder])

  if len(encoder.layers) > 0:
    nn = [tf.keras.layers.Flatten(input_shape=encoder.output_shape[1:])]
  else:
    nn = [tf.keras.layers.Flatten(input_shape=samples_shape[1:])]
  for i in range(0,len(ds_nn)):
    # nn.append(tf.keras.layers.Dense(ds_nn[i], kernel_regularizer=tf.keras.regularizers.l2()))
    nn.append(tf.keras.layers.Dense(ds_nn[i], kernel_initializer="glorot_uniform"))
    if i == len(ds_nn) - 1:
      nn.append(last_layer_nn_activation)
    else:
      nn.append(default_activation)
  nn = tf.keras.models.Sequential(nn)

  return autoencoder, encoder, decoder, nn


def mlp_entities(ds_encoder,ds_decoder,last_layer_decoder_activation,ds_nn,last_layer_nn_activation):
  encoder = []
  for i in range(1,len(ds_encoder)):
    encoder.append(tf.keras.layers.Dense(ds_encoder[i], input_shape=[ds_encoder[i-1]]))
    encoder.append(default_activation)
  encoder = tf.keras.models.Sequential(encoder)

  decoder = []
  for i in range(1,len(ds_decoder)):
    decoder.append(tf.keras.layers.Dense(ds_decoder[i], input_shape=[ds_decoder[i-1]]))
    if i == len(ds_decoder) - 1:
      decoder.append(last_layer_decoder_activation)
    else:
      decoder.append(default_activation)
  decoder = tf.keras.models.Sequential(decoder)

  if len(encoder.layers) == 0:
    autoencoder = tf.keras.models.Sequential([])
  else:
    autoencoder = tf.keras.models.Sequential([encoder, decoder])

  nn = []
  for i in range(1,len(ds_nn)):
    # nn.append(tf.keras.layers.Dense(ds_nn[i], input_shape=[ds_nn[i-1]], kernel_regularizer=tf.keras.regularizers.l2()))
    # nn.append(tf.keras.layers.Dense(ds_nn[i], input_shape=[ds_nn[i-1]]))
    nn.append(tf.keras.layers.Dense(ds_nn[i], input_shape=[ds_nn[i-1]], kernel_initializer="glorot_uniform"))
    if i == len(ds_nn) - 1:
      nn.append(last_layer_nn_activation)
    else:
      nn.append(default_activation)
  nn = tf.keras.models.Sequential(nn)

  return autoencoder, encoder, decoder, nn





def save(epoch=None):
  if epoch is None:  
    with open(os.path.join(directory,"clustering.pickle"), 'wb') as handle:
        pickle.dump(clustering, handle)
  else:
    if len(nn.layers) > 0:
      nn.save(os.path.join(directory,"nn.h5"))
    if len(autoencoder.layers) > 0:
      encoder.save(os.path.join(directory,"encoder.h5"))
      decoder.save(os.path.join(directory,"decoder.h5"))
      autoencoder.save(os.path.join(directory,"autoencoder.h5"))

    import pickle
    with open(os.path.join(directory,"dist_parameters.pickle"), 'wb') as handle:
        pickle.dump([logits.numpy(),
                    [locs[i].numpy() for i in range(number_of_dist)],
                    [tf.convert_to_tensor(scale_trils[i]).numpy() for i in range(number_of_dist)]], handle)

    with open(os.path.join(directory,"losses.pickle"), 'wb') as handle:
        pickle.dump(losses, handle)

    file = open(os.path.join(directory,"epoch.txt"),"w")
    file.write(str(epoch))
    file.close()


def cdist(A,B):
  # from https://stackoverflow.com/questions/43839120/compute-the-pairwise-distance-between-each-pair-of-the-two-collections-of-inputs

  p1 = tf.matmul(
      tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1),
      tf.ones(shape=(1, len(B)))
  )
  p2 = tf.transpose(tf.matmul(
      tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]),
      tf.ones(shape=(len(A), 1)),
      transpose_b=True
  ))

  res = (tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))

  return res


def nll(dist, samples):
    return -tf.reduce_mean(dist.log_prob(samples))

def mse(x,y):
  return tf.keras.losses.MeanSquaredError()(x,y)

@tf.function
def get_loss_and_grads(dist, nn, encoder, decoder, inputs, trainable_variables, lambdaa, cl_loss_type, var_features, var_locs, reg_logits, kl_loss):
    with tf.GradientTape() as tape:
        tape.watch(trainable_variables)
        z = encoder(inputs)
        reconstructed_inputs = decoder(z)
        x = nn(z)
        # kl_loss = 0
        # for i in range(len(dist.components)-1):
        #   for j in range(i+1,len(dist.components)):
        #     kl_loss += tfd.kl_divergence(dist.components[i],dist.components[j])
        # loss = nll(dist, x) + mse(inputs,reconstructed_inputs) -tf.reduce_mean(tf.math.reduce_std(x,axis=0))#-kl_loss + mse(tf.nn.softmax(logits), [1/number_of_dist for i in range(number_of_dist)]) -tf.reduce_mean(tf.math.reduce_std(x,axis=0))#-tf.reduce_mean(tf.math.reduce_std(locs,axis=0))
        # loss = 0
        # for i in range(len(locs)):
        #   loss += tf.reduce_mean(tf.sqrt(tf.reduce_sum((x - locs[i])**2,axis=1)))
        # loss /= len(locs)
        # # loss *= 100.0
        # loss += -tf.reduce_mean(tf.math.reduce_std(x,axis=0))
        loss = 0
        if cl_loss_type == "km":
          dist_matrix = cdist(x,locs)
          mask_clusters = tf.one_hot(tf.math.argmin(dist_matrix,axis=1),depth=dist_matrix.shape[1])
          # lambdaa =np.math.pow(0.01,_/epochs)
          loss+=tf.reduce_sum(dist_matrix * mask_clusters) / len(x) 
        elif cl_loss_type == "gmm":
          loss += nll(dist, x)
        else:
          raise ValueError("cl_loss_type must be either km or gmm")

        if var_features:
          loss += lambdaa*(-tf.reduce_mean(tf.math.reduce_std(x,axis=0)))
        
        if var_locs:
          loss += (-tf.reduce_mean(tf.math.reduce_std(locs,axis=0)))

        if reg_logits:
          loss += mse(tf.nn.softmax(logits), [1/number_of_dist for i in range(number_of_dist)])

        if kl_loss:
          kl_loss = 0
          for i in range(len(dist.components)-1):
            for j in range(i+1,len(dist.components)):
              kl_loss += tfd.kl_divergence(dist.components[i],dist.components[j])
          loss += (-kl_loss)

        # tf.print("km loss=",loss)
        # tf.print("std loss=",loss2)
        # loss+=lambdaa * loss2
        if len(nn.layers) > 0:
          loss+=mse(inputs,reconstructed_inputs)
        
        # tf.print(nn.trainable_variables)
        # tf.print(loss)
        # tf.print(dist_matrix)
        # tf.print(locs)
        # tf.print(x)
        
        grads = tape.gradient(loss, trainable_variables)
        # tf.print("grads:",tf.reduce_sum(grads[0]))
        # # tf.print("pre mask clusters:",tf.math.argmin(dist_matrix,axis=1),summarize=-1)
        # tf.print("dist_matrix FULL:",dist_matrix,summarize=-1)
        # for i in range(len(grads)):
        #   if tf.math.is_nan(tf.reduce_sum(grads[i])):
        #     tf.print("IS NAN")
        # tf.print("dist_matrix:",tf.reduce_sum(dist_matrix))
        # tf.print("make_clusters:",tf.reduce_sum(mask_clusters))
        # tf.print(x)
        # tf.print("loss1=",tf.reduce_sum(dist_matrix * mask_clusters) / len(x) )
        # tf.print("loss2=",loss2)
    return loss, grads

def predict_clustering(cl_loss_type):
  clustering = []
  if cl_loss_type == "gmm":
    dist_cat_log_probs = [dist.cat.log_prob(i) for i in range(number_of_dist)]
    for batch in dataset_not_shuffled:
      z = encoder(batch) 
      x = nn(z)
      dist_components_weigthed_log_probs = np.zeros((number_of_dist,len(batch)))
      for i in range(number_of_dist):
        dist_components_weigthed_log_probs[i,:] = dist_cat_log_probs[i] + dist.components[i].log_prob(x)
      clustering += list(np.argmax(dist_components_weigthed_log_probs, axis=0))
  elif cl_loss_type == "km":
    for batch in dataset_not_shuffled:
      z = encoder(batch)
      x = nn(z)
      dist_matrix = cdist(x,locs)
      mask_clusters = tf.math.argmin(dist_matrix,axis=1).numpy()
      clustering += list(mask_clusters)
  else:
    raise ValueError("cl_loss_type must be either km, or gmm")
  return clustering



if ae_type == "mlp":
  ds_encoder = [samples.shape[1]] + ds_encoder
  ds_nn = [ds_encoder[-1]] + ds_nn
  ds_decoder = list(reversed(ds_encoder))
  autoencoder, encoder, decoder, nn = mlp_entities(ds_encoder,ds_decoder,last_layer_decoder_activation,ds_nn,last_layer_nn_activation)
else:
  ds_decoder = list(reversed(ds_encoder))
  autoencoder, encoder, decoder, nn = cnn_entities(ds_encoder,ds_decoder,last_layer_decoder_activation,ds_nn,last_layer_nn_activation,samples.shape)

# loc=tf.Variable(tf.zeros([prob_d]), name='loc')
# scale_tril=tfp.util.TransformedVariable(
#                       tf.eye(prob_d, dtype=tf.float32),
#                       tfp.bijectors.FillScaleTriL(),
#                       name="raw_scale_tril")
# dist = tfd.MultivariateNormalTriL(
#         loc=loc,
#         scale_tril=scale_tril)

logits = tf.Variable(np.array([0.]*number_of_dist,dtype=np.float32), name="logits",trainable=logits_trainable)
# locs = [tf.Variable(np.random.rand(prob_d)*2.0-1.0, dtype=tf.float32, name='loc'+str(i)) for i in range(number_of_dist)]
# locs = [[1.0,1.0],[-1.0,-1.0],[1.0,-1.0]]
aux_loc = [-loc_inner_value for i in range(prob_d)]
locs = []
for i in range(prob_d):
  local_aux_loc = np.array(aux_loc)
  local_aux_loc[i] = -local_aux_loc[i]
  locs.append(local_aux_loc)
# locs_mean = np.mean(locs,axis=1)
# for i in range(prob_d):
#   locs[i] = locs[i] - locs_mean
# locs = [float(i) for i in range(number_of_dist)]
# locs = np.random.rand(number_of_dist,prob_d)*100.0
locs = [tf.Variable(locs[i], dtype=tf.float32, name='loc'+str(i),trainable=locs_trainable) for i in range(number_of_dist)]
# locs = [tf.Variable(locs[i], 
#                     dtype=tf.float32, name='loc'+str(i)) for i in range(number_of_dist)]

scale_trils = [tfp.util.TransformedVariable(
                      tf.eye(prob_d, dtype=tf.float32),
                      tfp.bijectors.FillScaleTriL(),
                      name="raw_scale_tril"+str(i),trainable=covs_trainable) for i in range(number_of_dist)]
if dist_type == "normal":
  dist = tfd.Mixture(
    cat=tfd.Categorical(logits=logits),
    components=[
      tfd.MultivariateNormalTriL(
          loc=locs[i],
          scale_tril=scale_trils[i]) for i in range(number_of_dist)
  ])
else:
  dist = tfd.Mixture(
    cat=tfd.Categorical(logits=logits),
    components=[
      tfd.MultivariateStudentTLinearOperator(
          df=1,
          loc=locs[i],
          scale=tf.linalg.LinearOperatorLowerTriangular(scale_trils[i])) for i in range(number_of_dist)
  ])


def main():
  if not os.path.isdir(directory):
    os.mkdir(directory)

  ds_encoder = ds_encoder[:encoder_depth]
  ds_nn = ds_nn[(len(ds_nn)-nn_depth):]
  if nn_depth == 0 and encoder_depth != 0:
    ds_encoder[-1] = prob_d
  
  last_layer_decoder_activation = tf.keras.layers.Activation(last_layer_decoder_activation)
  last_layer_nn_activation = tf.keras.layers.Activation(last_layer_nn_activation)
  default_activation = tf.keras.layers.Activation(default_activation)

  # def train(dist, autoencoder, encoder, decoder, nn, samples):
  losses = []
  # vars = []
  trainable_variables = dist.trainable_variables 
  if len(nn.layers) > 0:
    trainable_variables += tuple(nn.trainable_variables) 
  if len(autoencoder.layers) > 0:
    trainable_variables += tuple(autoencoder.trainable_variables)
  if len(encoder.layers) > 0:
    best_encoder = tf.keras.models.clone_model(encoder)
    # best_encoder.build((None, samples.shape[1]))
    best_encoder.set_weights(encoder.get_weights())
  if len(decoder.layers) > 0:
    best_decoder = tf.keras.models.clone_model(decoder)
    best_decoder.set_weights(decoder.get_weights())
  if len(autoencoder.layers) > 0:
    best_autoencoder = tf.keras.models.clone_model(autoencoder)
    best_autoencoder.set_weights(autoencoder.get_weights())
  best_dist = dist.copy()
  if len(nn.layers) > 0:
    best_nn = tf.keras.models.clone_model(nn)
    best_nn.set_weights(nn.get_weights())
  best_mean_loss = np.float32("inf")
  for _ in range(epochs):
      print(_)
      mean_loss = 0
      count = 0
      for batch in dataset:
        loss, grads = get_loss_and_grads(dist, nn, encoder, decoder, batch, trainable_variables, lambdaa, cl_loss_type, var_features, var_locs, reg_logits, kl_loss)
        # plt.figure()
        # plt.imshow(grads[0].numpy().reshape(28,-1))
        # plt.show()
        # print(loss)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        mean_loss += loss
        count+=1
        if np.isnan(loss):
          break
      mean_loss/=count
      losses.append(mean_loss)
      if mean_loss < best_mean_loss:
        if len(encoder.layers) > 0:
          best_encoder = tf.keras.models.clone_model(encoder)
          # best_encoder.build((None, samples.shape[1]))
          best_encoder.set_weights(encoder.get_weights())
        if len(decoder.layers) > 0:
          best_decoder = tf.keras.models.clone_model(decoder)
          best_decoder.set_weights(decoder.get_weights())
        if len(autoencoder.layers) > 0:
          best_autoencoder = tf.keras.models.clone_model(autoencoder)
          best_autoencoder.set_weights(autoencoder.get_weights())
        best_dist = dist.copy()
        if len(nn.layers) > 0:
          best_nn = tf.keras.models.clone_model(nn)
          best_nn.set_weights(nn.get_weights())

        save(_)
        best_mean_loss = mean_loss
      # nll_loss = train(dist, autoencoder, encoder, decoder, nn, samples)
      if plot_at_each_iteration:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Cost function')
        plt.show()

      output = nn(encoder(batch)).numpy()
      dist_matrix = cdist(output,locs)
      mask_clusters = tf.math.argmin(dist_matrix,axis=1).numpy()
      plt.figure()
      plt.scatter(output[:,0],output[:,1],c=mask_clusters)
      plt.show()

      # print(trainable_variables)
      # print("NN")
      if np.isnan(mean_loss): 
        break
      # print(nn.trainable_variables)
      # vars.append([x.numpy() for x in trainable_variables])
  #   return nll_loss

  if len(encoder.layers) > 0:
    encoder = tf.keras.models.clone_model(best_encoder)
    encoder.set_weights(best_encoder.get_weights())
  if len(decoder.layers) > 0:
    decoder = tf.keras.models.clone_model(best_decoder)
    decoder.set_weights(best_decoder.get_weights())
  if len(autoencoder.layers) > 0:
    autoencoder = tf.keras.models.clone_model(best_autoencoder)
    autoencoder.set_weights(best_autoencoder.get_weights())
  dist = best_dist.copy()
  if len(nn.layers) > 0:
    nn = tf.keras.models.clone_model(best_nn)
    nn.set_weights(best_nn.get_weights())

  # nll_loss = train(dist, autoencoder, encoder, decoder, nn, samples)
  plt.plot(losses)
  plt.xlabel('Epochs')
  plt.ylabel('Cost function')
  # trainable_variables
