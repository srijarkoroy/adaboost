import matplotlib.pyplot as plt
from src.datagenerate import dataset
from src.plot import plot_adaboost
from src.boosting import AdaBoost

X, y = dataset(100,10,2)

# assign our individually defined functions as methods of our classifier
clf = AdaBoost().fit(X, y, iters=50)  

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.4%}')

errors = list(clf.errors)
ada_error = list(clf.ada_errors)

y = []
for i in range(1,51):
  y.append(i)

# plotting the error curve (expected - exponentially decreasing)
plt.plot(y,errors)
plt.plot(y,ada_error)
plt.ylabel('error')
plt.xlabel('iteration')
plt.legend(['weak hypothesis error', 'final hypothesis error'], loc='upper right')
plt.show()
