from functions import *
from model import LSTMWithOutput, ManyToOneLSTM, loss_functions
import numpy as np

np.random.seed(12321)

def check_function_grad(func, h):
    shape = h.shape

    df_dh = np.zeros(shape)

    eps = 1e-7

    for i in range(shape[0]):
        h_plus_eps = h.copy()
        h_minus_eps = h.copy()

        h_plus_eps[i] += np.array(eps)
        h_minus_eps[i] += np.array(-eps)
        h_plus_loss = func(h_plus_eps)
        h_minus_loss = func(h_minus_eps)

        df_dh[i] = np.array((h_plus_loss - h_minus_loss) / (2*eps))
    
    return df_dh

def run_checks(checker, times, eps, name):
    results = np.array([np.linalg.norm(checker()) for _ in range(times)])
    result = (results < eps).all()
    if not result:
        print(results)
    else:
        print(f"{name} checks are passed")

def check_square():
    n_dims = 10

    x = np.random.rand(n_dims, 1)
    y = np.random.rand(n_dims, 1)
    dL_dx = square_loss_grad(x, y)
    dL_dx_app = check_function_grad(lambda x: square_loss(x, y), x)
    
    return dL_dx - dL_dx_app

def check_softmax_ce():
    n_dims = 10
    x = np.random.rand(n_dims, 1)
    y = np.zeros((n_dims, 1))
    y[np.random.randint(10)] = np.array([1])
    dL_dx = softmax_cross_entropy_loss_grad(x, y)
    dL_dx_app = check_function_grad(lambda x: softmax_cross_entropy_loss(x, y), x)
    
    return dL_dx_app - dL_dx



def check_linear_with_activation():
    n_dims_in = 7
    n_dims_out = 10

    x = np.random.rand(n_dims_in, 1)
    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    W1 = np.random.rand(n_dims_out, n_dims_in)
    b1 = np.random.rand(n_dims_out, 1)
    W2 = np.random.rand(n_dims_out, n_dims_out)
    b2 = np.random.rand(n_dims_out, 1)
    l = np.dot(W1, x) + b1
    l = np.dot(W2, l) + b2
    dL_dx = softmax_cross_entropy_loss_grad(l, y)
    
    dL_dx = np.dot(W2.T, dL_dx)
    dL_dx = np.dot(W1.T, dL_dx)
    dL_dx_app = check_function_grad(lambda x: softmax_cross_entropy_loss(np.dot(W2, np.dot(W1, x) + b1) + b2 , y), x)
    return dL_dx_app - dL_dx


class OneToOneLSTM(LSTMWithOutput):
    def forward(self, inp, h=None, c=None):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))

        c, h = self.forward_step(inp, c, h)

        y = self.linear_output(h)

        return y


def check_one_layer():
    n_dims_in = 7
    n_dims_hidden = 9
    n_dims_out = 4
    model = OneToOneLSTM(n_dims_in, n_dims_hidden, "softmax_ce", n_dims_out, softmax)
    
    x = np.random.rand(n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden , 1)
    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    model.initialize_gradients()
    model.enable_caching = False

    dL_dh_app = check_function_grad(lambda h: cross_entropy_loss(model.forward(x, h, c), y), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = model.backprop_output(y)
    dL_dh = model.backprop(dL_dh, 0)

    return dL_dh_app - dL_dh

def check_many_to_one():
    n_dims_in = 7
    n_dims_hidden = 9
    n_dims_out = 4
    n_samples = 20
    model = ManyToOneLSTM(n_dims_in, n_dims_hidden, "softmax_ce", n_dims_out, softmax)
    x = np.random.rand(n_samples, n_dims_in, 1)
    x = np.random.rand(n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden , 1)
    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    model.initialize_gradients()
    model.enable_caching = False

    dL_dh_app = check_function_grad(lambda h: cross_entropy_loss(model.forward(x, h, c), y), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = model.backprop_output(y)
    dL_dh = model.backprop(dL_dh, 0)

    return dL_dh_app - dL_dh


# check_one_layer()
run_checks(check_many_to_one, 10, 1e-7, "Many to One")
# check_activations()
# check_linear_with_activation()