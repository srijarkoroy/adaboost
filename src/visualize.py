import matplotlib.pyplot as plt
from datagenerate import dataset
from plot import plot_adaboost
from boosting import AdaBoost

def truncated_adaboost(clf, t: int):         #Truncate a fitted AdaBoost up to (and including) a particular iteration
  assert t > 0     #t must be a positive integer
  from copy import deepcopy
  new_clf = deepcopy(clf)
  new_clf.stumps = clf.stumps[:t]
  new_clf.stump_weights = clf.stump_weights[:t]
  return new_clf

def plot_iter_adaboost(X, y, clf, iters):       #Plot weak learner and cumulaive strong learner at each iteration.
   
  # larger grid
  fig, axes = plt.subplots(figsize=(8, iters*3),
                            nrows=iters,
                            ncols=2,
                            sharex=True,
                            dpi=100)
  fig.set_facecolor('white')

  _ = fig.suptitle('Decision boundaries after every iteration')
  for i in range(iters):
    ax1, ax2 = axes[i]

    # Plot weak learner
    _ = ax1.set_title(f'Weak learner at iteration {i + 1}')
    plot_adaboost(X, y, clf.stumps[i],
                  sample_weights=clf.sample_weights[i],
                  annotate=False, ax=ax1)

    # Plot strong learner
    trunc_clf = truncated_adaboost(clf, t = i + 1)
    _ = ax2.set_title(f'Strong learner at iteration {i + 1}')
    plot_adaboost(X, y, trunc_clf,
                  sample_weights=clf.sample_weights[i],
                  annotate=False, ax=ax2)

  plt.tight_layout()
  plt.subplots_adjust(top=0.95)
  plt.show()

X, y = dataset(100,10,2)
clf = AdaBoost().fit(X, y, iters=50)
plot_iter_adaboost(X, y, clf, 50)
