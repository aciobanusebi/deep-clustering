import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt
import numpy as np

from metric_functions import *
from module_functions import *
from get_datasets import *

dim_pca = None # None or a number; cannot use CNN with this "on"

n_runs = 1
epochs = 10

read_model_from_file = False
l2_reg_on_nn = False
data_name = "20news"
original_data_name = data_name
if dim_pca is not None:
    data_name+="PCA"
ae_type = "mlp" #@param ["mlp", "cnn"]
dist_type = "normal" #@param ["normal", "cauchy"]
encoder_depth = "0" #@param [0, 1, 2]
encoder_depth = int(encoder_depth)
nn_depth = "2" #@param [0, 1, 2, 3]
nn_depth = int(nn_depth)
lambdaa = 50.0 # [0,inf); for std
cl_loss_type = "km" # "km", "gmm"
id = f"{cl_loss_type}_{ae_type}_{dist_type}_{encoder_depth}_{nn_depth}_{int(lambdaa)}"
random_id = f"random_{id}"
print(id)
general_directory = "."

var_features = True # True/False
var_locs = False # True/False
reg_logits = False # True/False
kl_loss = False # True/False
logits_trainable = False # True/False
locs_trainable = False # True/False
covs_trainable = False # True/False
loc_inner_value = 1.0
# number_of_dist = 10
if original_data_name == "mnist":
    samples, real_labels = get_data_mnist()
elif original_data_name == "20news":
    samples, real_labels = get_data_20news()
elif original_data_name == "usps":
    samples, real_labels = get_data_usps()
    
number_of_dist = len(np.unique(real_labels))
print(number_of_dist)

import math
if ae_type == "cnn":
    samples = samples.reshape((-1,
                             int(math.sqrt(samples.shape[1])),
                             int(math.sqrt(samples.shape[1])),
                             1))
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

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)



if dim_pca is not None:
    import numpy as np
    from sklearn.decomposition import PCA
    X = samples
    pca = PCA(n_components=dim_pca)
    samples = pca.fit_transform(X)

    
random_mean_losses = []
random_clusterings = []
best_mean_losses = []
clusterings = []
for run in range(n_runs):
    print(run)
    random_mean_loss,random_clustering, best_mean_loss,clustering = main(read_model_from_file,
        run,
        samples,
        ae_type,
        dist_type,
        encoder_depth,
        nn_depth,
        id,
        random_id,
        data_name,
        general_directory,
        lambdaa,
        cl_loss_type,
        var_features,
        var_locs,
        reg_logits,
        kl_loss,
        logits_trainable,
        locs_trainable,
        covs_trainable,
        loc_inner_value,
        number_of_dist,
        BATCH_SIZE,
        SHUFFLE_BUFFER_SIZE,
        prob_d,
        last_layer_decoder_activation,
        last_layer_nn_activation,
        default_activation,
        plot_at_each_iteration,
        ds_encoder,
        ds_nn,
        optimizer,
        epochs,
        l2_reg_on_nn)
    
    random_mean_losses.append(random_mean_loss)
    random_clusterings.append(random_clustering)
    best_mean_losses.append(best_mean_loss)
    clusterings.append(clustering)

    
    
import numpy as np
random_index = np.argmin(random_mean_losses)
final_random_clustering = random_clusterings[random_index]
final_random_metrics = [purity_score(real_labels,final_random_clustering),
    adjusted_rand_score(real_labels,final_random_clustering),
    normalized_mutual_info_score(real_labels,final_random_clustering),
    accuracy(real_labels,final_random_clustering)]
random_path = os.path.join(general_directory,data_name,random_id,"final_random_metrics.txt")
f = open(random_path, "w")
f.writelines([str(x)+"\n" for x in final_random_metrics])
f.close()

import numpy as np
index = np.argmin(best_mean_losses)
final_clustering = clusterings[index]

final_metrics = [purity_score(real_labels,final_clustering),
    adjusted_rand_score(real_labels,final_clustering),
    normalized_mutual_info_score(real_labels,final_clustering),
    accuracy(real_labels,final_clustering)]
path = os.path.join(general_directory,data_name,id,"final_metrics.txt")
f = open(path, "w")
f.writelines([str(x)+"\n" for x in final_metrics])
f.close()

random_mean_losses = []
random_clusterings = []
best_mean_losses = []
clusterings = []
for run in range(n_runs):
    print(run)
    random_mean_loss,random_clustering, best_mean_loss,clustering = main(True,
        run,
        samples,
        ae_type,
        dist_type,
        encoder_depth,
        nn_depth,
        id,
        random_id,
        data_name,
        general_directory,
        lambdaa,
        cl_loss_type,
        var_features,
        var_locs,
        reg_logits,
        kl_loss,
        logits_trainable,
        locs_trainable,
        covs_trainable,
        loc_inner_value,
        number_of_dist,
        BATCH_SIZE,
        SHUFFLE_BUFFER_SIZE,
        prob_d,
        last_layer_decoder_activation,
        last_layer_nn_activation,
        default_activation,
        plot_at_each_iteration,
        ds_encoder,
        ds_nn,
        optimizer,
        epochs,
        l2_reg_on_nn)
    
    random_mean_losses.append(random_mean_loss)
    random_clusterings.append(random_clustering)
    best_mean_losses.append(best_mean_loss)
    clusterings.append(clustering)