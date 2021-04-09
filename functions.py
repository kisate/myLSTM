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

def softmax_cross_entropy_loss(z, y):
    e_z = np.exp(z - np.max(z))
    # if (np.sum(- np.log (np.sum(e_z*y, axis=0) / e_z.sum(axis=0)) * y, axis=0) == np.nan).any():
    # print((np.sum(e_z*y, axis=0) / e_z.sum(axis=0)) * y)
    return np.sum(- np.ma.log (np.sum(e_z*y, axis=0) / e_z.sum(axis=0)).filled(0) * y, axis=0)

def softmax_cross_entropy_loss_grad(z, y):
    e_z = np.exp(z - np.max(z))
    s = e_z.sum(axis=0)
    return y * (np.sum(e_z*y, axis=0) / s - 1) + e_z*(1 - y) / s

def square_loss(z, y):
    return np.sum(0.5 * np.square(y - z), axis=0)

def square_loss_grad(z, y):
    return z - y