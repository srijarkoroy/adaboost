import numpy as np
from sklearn.datasets import make_gaussian_quantiles

def dataset(n,random_seed):
  if random_seed:
    np.random.seed(random_seed)
    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)
    return X, y*2-1