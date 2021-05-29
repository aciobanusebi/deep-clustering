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
  - deep_distributions_MAIN.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN.ipynb): main.py in a .ipynb file
  - deep_distributions_MAIN gmm normal.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20gmm%20normal.ipynb): deep_distributions_MAIN.ipynb with the *gmm1* setup from the paper
  - deep_distributions_MAIN gmm cauchy.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20gmm%20cauchy.ipynb): deep_distributions_MAIN.ipynb with the *cmm1* setup from the paper
  - deep_distributions_MAIN km1.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20km1.ipynb): deep_distributions_MAIN.ipynb with the *km1* setup from the paper
  - deep_distributions_MAIN km2.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/deep-clustering/blob/main/deep_distributions_MAIN%20km2.ipynb): deep_distributions_MAIN.ipynb with the *km50* setup from the paper
