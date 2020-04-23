
import numpy as np
import copy

np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# extended data
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)



print(X)

def linear(W, X):
    """return (c, m)"""
    return np.dot(W.T, X)  # Z


def sigmoid(Z):
    """return (c, m)"""
    return 1 / (1 + np.exp(-Z))  # A


def grad(W, X, Y):
    Z = linear(W, X)
    A = sigmoid(Z)
    # return np.multiply((A-Y), X)
    return np.multiply((A - Y), X) + (1e-4) * W


def fit(docs_train_vector, labels_train):
    list_label = list(set(labels_train))
    number_label = len(list_label)
    number_atr = docs_train_vector.shape[0]
    number_doc = docs_train_vector.shape[1]
    max_count = 10000
    learning_rate = 0.05
    wi = np.random.randn(number_atr, 1)
    count = 0
    while count < max_count:
        index = np.random.permutation(number_doc)
        for i in index:
            xi = docs_train_vector[:, i].reshape(number_atr, 1)
            yi = labels_train[i]
            z = np.dot(wi.T, xi)
            zi = sigmoid(z)
            w_new = wi - learning_rate * (zi - yi) * xi
            count += 1

            # if count % check_w_after == 0:
            #   if np.linalg.norm(w_new - wi) < tol:
            #      wi = copy.deepcopy(w_new)
            #     break
            wi = copy.deepcopy(w_new)
    # self.w = w0
    return wi

w1 = fit(X,y)
Y=sigmoid(np.dot(w1.T, X))
Y = list(Y[0])
print(Y)
for i in range(len(Y)):
    if Y[i] > 0.5:
        Y[i] = 1
    else:
        Y[i] = 0
cnt = 0
for i in range(len(Y)):
    if Y[i] == y[i]:
        cnt +=1
print(cnt/len(Y))