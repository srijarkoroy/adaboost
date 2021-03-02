import matplotlib.pyplot as plt
from datagenerate import dataset
from plot import plot_adaboost
from boosting import AdaBoost

X, y = dataset(100,10,2)

clf = AdaBoost().fit(X, y, iters=50)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.4%}')

errors = list(clf.errors)
ada_error = list(clf.ada_errors)

y = []
for i in range(1,51):
  y.append(i)

plt.plot(y,errors)
plt.plot(y,ada_error)
plt.ylabel('error')
plt.xlabel('iteration')
plt.legend(['weak hypothesis error', 'final hypothesis error'], loc='upper right')
plt.show()