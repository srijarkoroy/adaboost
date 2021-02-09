from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_adaboost(X: np.ndarray,               #Plot ± samples in 2D, optionally with decision boundary
                  y: np.ndarray,
                  clf=None,
                  sample_weights: Optional[np.ndarray] = None,
                  annotate: bool = False,
                  ax: Optional[mpl.axes.Axes] = None) -> None:

  assert set(y) == {-1, 1}                 

  if not ax:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    fig.set_facecolor('white')

  pad = 1
  x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
  y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

  if sample_weights is not None:
    sizes = np.array(sample_weights) * X.shape[0] * 100
  else:
    sizes = np.ones(shape=X.shape[0]) * 100

  X_pos = X[y == 1]
  sizes_pos = sizes[y == 1]
  ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='green')

  X_neg = X[y == -1]
  sizes_neg = sizes[y == -1]
  ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

  if clf:
    plot_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                          np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if list(np.unique(Z)) == [1]:
      fill_colors = ['g']
    else:
      fill_colors = ['b', 'g']

    ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)

  if annotate:
    for i, (x, y) in enumerate(X):
      offset = 0.05
      ax.annotate(f'$x_{i + 1}$', (x + offset, y - offset))

  ax.set_xlim(x_min+0.5, x_max-0.5)
  ax.set_ylim(y_min+0.5, y_max-0.5)
  ax.set_xlabel('$x_1$')
  ax.set_ylabel('$x_2$')