# Fixed Representatives and Variable Features: From Regularized Deep Clustering to Classic Clustering

# Paper abstract

The clustering problem in machine learning is often tackled using classic/traditional models, like k-means and mixtures of Gaussians, but in the past few years, the deep learning community has also developed algorithms for unsupervised learning and for clustering specifically. A very common pattern in these new algorithms is to map the data into a new space via a neural network and at the same time, to perform clustering with these new data representations. This general idea comes with a cost: in order to avoid trivial/unsatisfactory solutions one should regularize the model. We started from this framework and simplified the model (as opposed to adding complex regularization schemes) such that a clustering is learnable, i.e. we do not obtain trivial solutions. We retained a multilayer perceptron and added a simple regularization composed of two characteristics of the new space: it has fixed representatives, e.g. the centroids for k-means or the distributions for the mixture of Gaussians are fixed, and the features have high variance. We compared the new model to classic and deep clustering algorithms on three datasets, and the empirical results show that the proposed method is close to the classic ones in terms of performance.

# Index words
Machine Learning, Clustering, Deep Clustering, k-means, Mixture of Gaussians, Regularization

# Code

## .h5 files
- usps.h5: USPS dataset from https://www.kaggle.com/bistaumanga/usps-dataset

## .py files
- get_datasets.py: functions for fetching the following datasets
  - 20news
  - MNIST
  - USPS
- main.py: the code that a user should use in order to apply our methods
- metric_functions.py: functions for computing 4 clustering evaluation metrics
- module_functions.py: a core function with inner functions in which we implement our algorithms

## .ipynb files
- clustering_*
  - clustering_exps_20news.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/clustering_exps_20news.ipynb): random, k-means, and EM/GMM clustering on the 20news dataset
  - clustering_exps_20news_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/clustering_exps_20news_PCA.ipynb): random, k-means, and EM/GMM clustering on the first 100 principal components of the 20news dataset
  - clustering_exps_mnist.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/clustering_exps_mnist.ipynb): random, k-means, and EM/GMM clustering on the MNIST dataset
  - clustering_exps_mnist_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/clustering_exps_mnist_PCA.ipynb): random, k-means, and EM/GMM clustering on the first 100 principal components of the MNIST dataset
  - clustering_exps_usps.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/clustering_exps_usps.ipynb): random, k-means, and EM/GMM clustering on the USPS dataset
  - clustering_exps_usps_PCA.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/clustering_exps_usps_PCA.ipynb): random, k-means, and EM/GMM clustering on the first 100 principal components of the USPS dataset
- **deep_distributions_***
  - **deep_distributions_MAIN.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN.ipynb): main.py in a .ipynb file
  - **deep_distributions_MAIN gmm normal.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20gmm%20normal.ipynb): deep_distributions_MAIN.ipynb with the *gmm1* setup from the paper
  - **deep_distributions_MAIN gmm cauchy.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20gmm%20cauchy.ipynb): deep_distributions_MAIN.ipynb with the *cmm1* setup from the paper
  - **deep_distributions_MAIN km1.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20km1.ipynb): deep_distributions_MAIN.ipynb with the *km1* setup from the paper
  - **deep_distributions_MAIN km2.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20km2.ipynb): deep_distributions_MAIN.ipynb with the *km50* setup from the paper

## deep_distributions* hyperparameters
- **dim_pca**: 
  - None or a number; 
  - how many PCs to retain; 
  - cannot use CNN with this "on"
- **n_runs**: 
  - number; 
  - how many independent runs of the model to try
- **epochs**: 
  - number; 
  - number of epochs for one run
- **read_model_from_file**: 
  - True/False; 
  - read the model from file?
- **l2_reg_on_nn**: 
  - True/False; 
  - use L2 regularization on nn?
- **data_name**: 
  - "20news"/"mnist"/"usps"
- **ae_type**: 
  - "mlp"/"cnn"; 
  - "cnn" works only for "mnist", "usps"
- **dist_type**: 
  - "normal"/"cauchy"; 
  - the type of distribution if a mixture model is used; it is ignored if k-means is used
- **encoder_depth**: 
  - 0/1/2; 
  - how many layers the encoder contains (the decoder is mirrored)
- **nn_depth**: 
  - 0/1/2/3; 
  - how many layers nn contains
- **lambdaa**: 
  - positive real number; 
  - <img src="https://latex.codecogs.com/gif.latex?\lambda_V"/> in the paper; 
  - the importance of the variability loss
- **cl_loss_type**: 
  - "km"/"gmm"; 
  - km stands for k-means and gmm for a mixture model (either gmm or cmm)
- **general_directory**: 
  - the models will be saved here
- **var_features**: 
  - True/False; 
  - if True, the "variable features" loss (<img src="https://latex.codecogs.com/gif.latex?\text{loss}_V"/>) is added to the clustering loss
- **var_locs**: 
  - True/False; 
  - if True, the "variable <img src="https://latex.codecogs.com/gif.latex?\mu"/>s" loss is added to the clustering loss
- **reg_logits**: 
  - True/False;
  - if True, a regularization loss for logits (the <img src="https://latex.codecogs.com/gif.latex?\pi"/> probabilities should be close to the uniform distribution) is added to the clustering loss
- **kl_loss**: 
  - True/False;
  - if True, a KL between each distribution in the mixture is added to the clustering loss
- **logits_trainable**: 
  - True/False; 
  - if True, <img src="https://latex.codecogs.com/gif.latex?\pi"/>s are trainable
- **locs_trainable**: 
  - True/False; 
  - if True, <img src="https://latex.codecogs.com/gif.latex?\mu"/>s are trainable
- **covs_trainable**: 
  - True/False; 
  - if True, <img src="https://latex.codecogs.com/gif.latex?\Sigma"/>s are trainable
- **loc_inner_value**: 
  - a number; 
  - <img src="https://latex.codecogs.com/gif.latex?\mu"/>s initialization is multiplied by this value; 
  - if set to None, then we use uniform random init in [0,1]; 
  - we used this random init when - starting the project
- **BATCH_SIZE**: 
  - number; 
  - batch size
- **SHUFFLE_BUFFER_SIZE**: 
  - number; 
  - shuffle buffer size
- **prob_d**: 
  - number; 
  - the output dimensionality
- **last_layer_decoder_activation**: 
  - 'sigmoid'/'linear'/'relu'/'tanh'; 
  - the output activation in the autoencoder
- **last_layer_nn_activation**: 
  - 'sigmoid'/'linear'/'relu'/'tanh'; 
  - the output activation in nn
- **default_activation**: 
  - 'sigmoid'/'linear'/'relu'/'tanh'; 
  - the hidden activations in the autoencoder and nn
- **plot_at_each_iteration**: 
  - True/False; 
  - if True, useful plots are visualized at each iteration
- **ds_encoder**: 
  - list of numbers; 
  - a list which gives the number of hidden units for MLP and filters for CNN in each layer of the encoder (the decoder is mirrored)
- **ds_nn**: 
  - list of numbers; 
  - a list which gives the number of hidden units in each layer of nn
- **optimizer**: 
  - a tf.keras optimizer

## Supplementary observations on the MNIST dataset with 10 clusters
- if the batch_size is equal to the number of images (70.000), then the results are around those (not the same!) obtained with the initialization (Init model) even if we use the variability <img src="https://latex.codecogs.com/gif.latex?\text{loss}_V"/> or not
- when using gmm1, an autoencoder with a depth-2 encoder can improve the results (0.59, 0.40, 0.49, 0.59);
  - the same model but without nn returns (0.58, 0.38, 0.50, 0.56)
- when using km50, an autoencoder with a depth-2 encoder can worsen the results (0.40, 0.20, 0.33, 0.38);
  - the same model but without nn returns (0.35, 0.26, 0.40, 0.35)
