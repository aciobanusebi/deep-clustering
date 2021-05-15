import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

print("TF version:", tf.__version__)
print("TFP version:", tfp.__version__)

import matplotlib.pyplot as plt
import numpy as np

from metric_functions import *
from module_functions import *

ae_type = "mlp" #@param ["mlp", "cnn"]
dist_type = "normal" #@param ["normal", "cauchy"]
encoder_depth = "0" #@param [0, 1, 2]
encoder_depth = int(encoder_depth)
nn_depth = "2" #@param [0, 1, 2, 3]
nn_depth = int(nn_depth)

id = f"{ae_type}_{dist_type}_{encoder_depth}_{nn_depth}"
print(id)

general_directory = "E:/paper-2021/mixtures"
directory = os.path.join(general_directory,id)

lambdaa = 50.0 # [0,inf); for std
cl_loss_type = "km" # "km", "gmm"
var_features = True # True/False
var_locs = False # True/False
reg_logits = False # True/False
kl_loss = False # True/False
logits_trainable = False # True/False
locs_trainable = False # True/False
covs_trainable = False # True/False
loc_inner_value = 1.0
number_of_dist = 10
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024
prob_d = number_of_dist
last_layer_decoder_activation = 'sigmoid'
last_layer_nn_activation = 'linear'
default_activation = 'tanh'
plot_at_each_iteration = True
if ae_type == "mlp":
  ds_encoder = [128,64] # [64,128]/[64] for "cnn"; [increasing] for "mlp"
else:
  ds_encoder = [64,128]
ds_nn = [32,prob_d]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # learning_rate=0.00001; era 0.001
epochs = 10

main()

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)