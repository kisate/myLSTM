from functions import *
from model import *
import numpy as np

# np.random.seed(2312)

def check_function_grad(func, h):
    shape = h.shape

    df_dh = np.zeros(shape)

    eps = 1e-7

    for x in range(shape[0]):
        for y in range(shape[1]):
            h_plus_eps = h.copy()
            h_minus_eps = h.copy()

            h_plus_eps[x, y] += np.array(eps)
            h_minus_eps[x, y] += np.array(-eps)
            h_plus_loss = func(h_plus_eps)
            h_minus_loss = func(h_minus_eps)

            df_dh[x, y] = np.array((h_plus_loss - h_minus_loss) / (2*eps))
    
    return df_dh

# def check_param_grads(func, model: BaseLSTM):
#     results = []
#     for param in param_names:
#         W, U, b = model.params["param"]



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
    dL_dc = np.zeros(h.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, 0)

    return dL_dh_app - dL_dh

def check_many_to_one():
    n_dims_in = 7
    n_dims_hidden = 9
    n_dims_out = 4
    n_samples = 2

    model = ManyToOneLSTM(n_dims_in, n_dims_hidden, "square", n_dims_out, lambda x: x)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden , 1)

    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    model.initialize_gradients()
    model.enable_caching = False

    dL_dh_app = check_function_grad(lambda h: square_loss(model.forward(x, h, c), y), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = model.backprop_output(y)
    dL_dc = np.zeros(h.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, 0)

    return dL_dh_app - dL_dh

def check_many_to_many():
    n_dims_in = 10
    n_dims_hidden = 13
    n_dims_out = 7
    n_samples = 10

    model = ManyToManyLSTM(n_dims_in, n_dims_hidden, "square", n_dims_out, lambda x : x)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.zeros(h.shape)

    # y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])
    y = np.random.rand(n_samples, n_dims_out, 1)

    model.initialize_gradients()
    model.enable_caching = False

    # print(cross_entropy_loss(model.forward(x, h, c), y).sum())

    dL_dh_app = check_function_grad(lambda h: square_loss(model.forward(x, h, c), y).sum(), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = np.zeros(h.shape)
    dL_dc = np.zeros(h.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, 0, list(y))

    # print(dL_dh)
    # print(dL_dh_app)

    return dL_dh_app - dL_dh

# check_one_layer()
# run_checks(check_softmax_ce, 10, 1e-7, "softmax_ce")
run_checks(check_one_layer, 10, 1e-7, "One to One")
run_checks(check_many_to_one, 10, 1e-7, "Many to One")
run_checks(check_many_to_many, 10, 1e-7, "Many to Many")
# check_activations()
# check_linear_with_activation()