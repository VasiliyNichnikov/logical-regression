import random
import time

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from algorithm import LogisticRegression

data = load_digits()

X = data['data']
y = (data['target'] >= 5).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size=0.77,
                                                    shuffle=True)


m, n = X_train.shape
X_train = [list(item) for item in X_train]
X_test = [list(item) for item in X_test]
y_train = list(y_train)
y_test = list(y_test)


random.seed(13)
start = time.time()


logreg = LogisticRegression(n, X_test, y_test)
logreg.train(X_train, y_train, epochs=500)
end = time.time()

print(f"Time: {end - start}")

domain = np.arange(0, len(logreg.losses_train)) * logreg.report_current_results
plt.plot(domain, logreg.losses_train, label='Train')
plt.plot(domain, logreg.losses_test, label='Test')
plt.xlabel('Epoch number')
plt.ylabel('LogLoss')
plt.legend()
plt.show()


test_prediction = np.array(logreg.predict(X_test))
test_accuracy = np.sum((test_prediction > 0.5) == y_test) / len(test_prediction)
print(f'Точность на тестовой выборке: {round(test_accuracy * 100, 2)}%')

train_prediction = np.array(logreg.predict(X_train))
train_accuracy = np.sum((train_prediction > 0.5) == y_train) / len(train_prediction)
print(f'Точность на обученной выборке: {round(train_accuracy * 100, 2)}%')
