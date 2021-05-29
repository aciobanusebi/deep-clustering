def get_data_20news():
  import tensorflow as tf
  import numpy as np
  from sklearn.datasets import fetch_20newsgroups
  from sklearn.feature_extraction.text import TfidfVectorizer

  _20news = fetch_20newsgroups(subset="all")
  data = _20news.data
  target = _20news.target

  vectorizer = TfidfVectorizer(max_features=2000)
  data = vectorizer.fit_transform(data)
  data = data.toarray().astype(np.float32)

  return data, target

def get_data_mnist(k=10):
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()

  x_train = np.concatenate((x_train,x_test))
  y_train = np.concatenate((y_train,y_test))

  real_labels = y_train

  # # indices = np.isin(y_train,range(number_of_dist))
  # x_train = x_train[indices]
  # y_train = y_train[indices]

  samples = (x_train.reshape((x_train.shape[0],-1))/255.).astype(np.float32)
  
  indices = real_labels < k
    
  return samples[indices], real_labels[indices]

def get_data_usps():
  import h5py
  import numpy as np
  path = "./usps.h5"
  with h5py.File(path, 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]

  samples = np.concatenate((X_tr,X_te))
  real_labels = np.concatenate((y_tr,y_te))
  return samples, real_labels