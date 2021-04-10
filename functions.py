import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def sigmoid_grad(dL, z):
    a = sigmoid(z)

    return dL * a * (1 - a)

def tanh_grad(dL, z):
    a = tanh(z)

    return dL * (1 - np.square(a))

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def softmax_grad(dL, z):
    n = z.shape[0]
    s = softmax(z)
    res = []
    for i in range(n):
        i_vec = np.eye(n)[i].reshape(s.shape)
        line = np.sum(s*i_vec)*(i_vec * (1 - s) + (i_vec - 1)*s)
        res.append([np.sum(line * dL)])

    return np.array(res)

def softmax_cross_entropy_loss(z, y):
    e_z = np.exp(z - np.max(z, axis=0))
    return np.sum(- np.ma.log (np.sum(e_z*y, axis=0) / e_z.sum(axis=0)).filled(0) * y, axis=0)

def softmax_cross_entropy_loss_grad(z, y):
    e_z = np.exp(z - np.max(z, axis=0))
    s = e_z.sum(axis=0)
    return y * (np.sum(e_z*y, axis=0) / s - 1) + e_z*(1 - y) / s

def cross_entropy_loss(z, y):
    return -np.sum(np.ma.log(z*y).filled(0) * y, axis=0)

def square_loss(z, y):
    return np.sum(0.5 * np.square(y - z), axis=0)

def square_loss_grad(z, y):
    return z - y
