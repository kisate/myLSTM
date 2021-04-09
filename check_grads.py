from functions import *
import numpy as np

def check_function_grad(func, h):
    shape = h.shape

    df_dh = np.zeros(shape)

    eps = 1e-7

    for i in range(shape[0]):
        h_plus_eps = h.copy()
        h_minus_eps = h.copy()
        # print(original)
        
        h_plus_eps[i] += np.array(eps)
        h_minus_eps[i] += np.array(-eps)
        h_plus_loss = func(h_plus_eps)
        h_minus_loss = func(h_minus_eps)
        # print([(self.forward(x, h_plus_eps)[0] - self.forward(x, h_minus_eps)[0]) / (2*eps)])
        df_dh[i] = np.array((h_plus_loss - h_minus_loss) / (2*eps))
    
    return df_dh

def check_activations():
    n_dims = 10

    # for _ in range(10):
    #     x = np.random.rand(n_dims, 1)
    #     y = np.random.rand(n_dims, 1)
    #     dL_dx = square_loss_grad(x, y)
    #     dL_dx_app = check_function_grad(lambda x: square_loss(x, y), x)
    #     print(np.linalg.norm(dL_dx_app - dL_dx) < 1e-7)

    for _ in range(10):
        x = np.random.rand(n_dims, 1)
        y = np.zeros((n_dims, 1))
        y[np.random.randint(10)] = np.array([1])
        dL_dx = softmax_cross_entropy_loss_grad(x, y)
        dL_dx_app = check_function_grad(lambda x: softmax_cross_entropy_loss(x, y), x)
        # print(dL_dx_app)
        # print(dL_dx)
        print(np.linalg.norm(dL_dx_app - dL_dx) < 1e-7)


check_activations()