from functions import *

import numpy as np

param_names = [
    "f", "i", "g", "o"
]


activation_functions = {
    "sigmoid" : (sigmoid, sigmoid_grad),
    "tanh" : (tanh, tanh_grad)
}

loss_functions = {
    "square" : (square_loss, square_loss_grad),
    "softmax_ce" : (softmax_cross_entropy_loss, softmax_cross_entropy_loss_grad)
}

class BaseLSTM:
    def __init__(self, n_dims_in, n_dims_hidden, loss_func):
        self.n_dims_in = n_dims_in
        self.n_dims_hidden = n_dims_hidden
        self.loss_func = loss_func
        self.dL_dc = np.zeros((n_dims_hidden, 1))
        
        self.params = {}
        self.grads = {}
        
        for param_name in param_names:
            W = np.random.rand(n_dims_hidden, n_dims_in) * 0.01
            U = np.random.rand(n_dims_hidden, n_dims_hidden) * 0.01
            b = np.random.rand(n_dims_hidden, 1) * 0.01
            self.params[param_name] = (W, U, b)
        
        self.cache = []
        self.enable_caching = True

    def linear_activation_forward(self, x, h, param_name, func_name):
        W, U, b = self.params[param_name]
        activation_function = activation_functions[func_name][0]

        z = np.dot(W, x) + np.dot(U, h) + b
        a = activation_function(z)

        self.save_to_cache((param_name, x, h, z, a, func_name))
        return a

    def backprop_step_linear(self, dL_da):
        param_name, x, h, z, a, func_name = self.cache.pop()
        W, U, b = self.params[param_name]
        activation_function_grad = activation_functions[func_name][1]

        dL_dz = activation_function_grad(dL_da, z)
        
        dL_dW = np.dot(dL_dz, x.T)
        
        dL_dU = np.dot(dL_dz, h.T)
        dL_dh = np.dot(U.T, dL_dz)

        dL_db = np.sum(dL_dz, axis=1, keepdims=True) 

        W_grad, U_grad, b_grad = self.grads[param_name]
        
        self.grads[param_name] = (W_grad + dL_dW, U_grad + dL_dU, b_grad + dL_db)

        return dL_dh

    def backprop_step_no_output(self, dL_dh, dL_dc, cache):
        f, i, g, o, c_new, c = cache

        dL_do = tanh(c_new)*dL_dh

        dL_dc = dL_dc + tanh_grad(dL_dh * o, c_new)
        
        dL_df = dL_dc * c
        dL_di = dL_dc * g 
        dL_dg = dL_dc * i

        do_dh = self.backprop_step_linear(dL_do)
        dg_dh = self.backprop_step_linear(dL_dg)
        di_dh = self.backprop_step_linear(dL_di)
        df_dh = self.backprop_step_linear(dL_df)

        dL_dh = do_dh + df_dh + dg_dh + di_dh 
        dL_dc = dL_dc*f

        return dL_dh, dL_dc

    def backprop_step(self, dL_dh, dL_dc, data=None):
        h, cache = self.cache.pop()
        dL_dh, dL_dc = self.backprop_step_no_output(dL_dh, dL_dc, cache)
        return dL_dh, dL_dc

    
    def backprop(self, dL_dh, dL_dc, learning_rate, data_stack=None):

        while self.cache:
            if data_stack is None:
                dL_dh, dL_dc = self.backprop_step(dL_dh, dL_dc)
            else:
                dL_dh, dL_dc = self.backprop_step(dL_dh, dL_dc, data_stack.pop())
                

        for param_name, param in self.params.items():
            grads = self.grads[param_name]
            self.params[param_name] = tuple(param[i] - grads[i]*learning_rate for i in range(len(grads)))

        return dL_dh, dL_dc

    def forward_step(self, x, c, h):
        f,i,g,o = 0, 0,0,0
        f = self.linear_activation_forward(x, h, "f", "sigmoid")
        i = self.linear_activation_forward(x, h, "i", "sigmoid")
        g = self.linear_activation_forward(x, h, "g", "tanh")
        o = self.linear_activation_forward(x, h, "o", "sigmoid")

        c_new = f * c + i * g
        h_new = o * tanh(c_new)

        self.save_to_cache((h_new, (f, i, g, o, c_new, c)))
        
        return c_new, h_new

    def initialize_gradients(self):
        for param_name in param_names:
            W_grad = np.zeros((self.n_dims_hidden, self.n_dims_in))
            U_grad = np.zeros((self.n_dims_hidden, self.n_dims_hidden))
            b_grad = np.zeros((self.n_dims_hidden, 1))
            self.grads[param_name] = (W_grad, U_grad, b_grad)

    def save_to_cache(self, value):
        if (self.enable_caching):
            self.cache.append(value)

    def forward(self, inp, h=None, c=None):
        raise NotImplementedError()
    
    def train_on_example(self, x, y, learning_rate):
        raise NotImplementedError()


class LSTMWithOutput(BaseLSTM):
    def __init__(self, n_dims_in, n_dims_hidden, loss_func, n_dims_out, output_activation):
        super().__init__(n_dims_in, n_dims_hidden, loss_func)
        self.n_dims_out = n_dims_out
        self.output_activation = output_activation
        
        LW = np.random.rand(n_dims_out, n_dims_hidden) * 0.01
        Lb = np.random.rand(n_dims_out, 1) * 0.01

        self.params["L"] = (LW, Lb)

    def linear_output(self, x):
        LW, Lb = self.params["L"]

        y = np.dot(LW, x) + Lb

        self.save_to_cache((LW, Lb, x, y))

        return self.output_activation(y)

    def backprop_output(self, y):
        LW, Lb, h, y_out = self.cache.pop()

        dL_dy = loss_functions[self.loss_func][1](y_out, y)

        dL_dLW = np.dot(dL_dy, h.T)

        dL_db = np.sum(dL_dy, axis=1, keepdims=True)

        W_grad, b_grad = self.grads["L"]
        
        self.grads["L"] = (W_grad + dL_dLW, b_grad + dL_db)

        return np.dot(LW.T, dL_dy)

    def initialize_gradients(self):
        super().initialize_gradients()

        LW_grad = np.zeros((self.n_dims_out, self.n_dims_hidden))
        Lb_grad = np.zeros((self.n_dims_out, 1))
        self.grads["L"] = (LW_grad, Lb_grad)


class ManyToOneLSTM(LSTMWithOutput):
    def forward(self, inp, h=None, c=None):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
            
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))

        for x in inp:
            c, h = self.forward_step(x, c, h)

        y = self.linear_output(h)

        return y

class ManyToManyLSTM(LSTMWithOutput):
    def __init__(self, n_dims_in, n_dims_hidden, loss_func, n_dims_out, output_activation):
        super().__init__(n_dims_in, n_dims_hidden, loss_func, n_dims_out=n_dims_out, output_activation=output_activation)

    def forward(self, inp, h=None, c=None):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))
        
        res = []

        for x in inp:
            c, h = self.forward_step(x, c, h)
            y = self.linear_output(h)
            res.append(y)

        return np.array(res)

    def backprop_step(self, dL_dh, dL_dc, data):
        return super().backprop_step(dL_dh + self.backprop_output(data), dL_dc)
