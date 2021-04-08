import numpy as np

param_names = [
    "f", "i", "o", "g"
]

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

def square_loss(z, y):
    return np.sum(0.5 * np.square(y - z), axis=0)

def square_loss_grad(z, y):
    return z - y

activation_functions = {
    "sigmoid" : (sigmoid, sigmoid_grad),
    "tanh" : (tanh, tanh_grad)
}

class BaseLSTM:
    def __init__(self, n_dims_in, n_dims_hidden, n_dims_out):
        self.n_dims_in = n_dims_in
        self.n_dims_hidden = n_dims_hidden
        self.n_dims_out = n_dims_out
        
        self.params = {}
        self.grads = {}
        
        for param_name in param_names:
            W = np.random.rand(n_dims_hidden, n_dims_in) * 0.01
            U = np.random.rand(n_dims_hidden, n_dims_hidden) * 0.01
            b = np.random.rand(n_dims_hidden, 1) * 0.01
            self.params[param_name] = (W, U, b)
        
        LW = np.random.rand(n_dims_out, n_dims_hidden) * 0.01
        Lb = np.random.rand(n_dims_out, 1) * 0.01

        self.params["L"] = (LW, Lb)

        self.cache = []

    def linear_activation_forward(self, x, h, param_name, func_name):
        W, U, b = self.params[param_name]
        activation_function = activation_functions[func_name][0]

        # print(W, x)

        z = np.dot(W, x) + np.dot(U, h) + b
        a = activation_function(z)

        self.cache.append((param_name, x, h, z, a, func_name))
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

    def backprop_step_no_output(self, dL_dh, cache):
        f, i, g, o, c, c_prev = cache
        dL_do = tanh(c)

        c_grad = o*tanh_grad(dL_dh, c)

        dL_df = c_grad * c_prev * dL_do
        dL_di = c_grad * g * dL_do
        dL_dg = c_grad * i * dL_do
        
        dg_dh = self.backprop_step_linear(dL_dg)
        do_dh = self.backprop_step_linear(dL_do)
        di_dh = self.backprop_step_linear(dL_di)
        df_dh = self.backprop_step_linear(dL_df)

        return do_dh + df_dh + dg_dh + di_dh 

    def backprop_step(self, dL_dh, y=None):
        h, cache = self.cache.pop()
        dL_dh = self.backprop_step_no_output(dL_dh, cache)
        return dL_dh

    
    def backprop(self, dL_dh, learning_rate, y=None):

        while self.cache:
            if y is None:
                dL_dh = self.backprop_step(dL_dh)
            else:
                dL_dh = self.backprop_step(dL_dh, y.pop())
                

        for param_name in param_names:
            W, U, b = self.params[param_name]
            W_grad, U_grad, b_grad = self.params[param_name]

            self.params[param_name] = (W - W_grad*learning_rate, U - U_grad*learning_rate, b - b_grad*learning_rate)

        LW, Lb = self.params["L"]
        LW_grad, Lb_grad = self.grads["L"]

        self.params["L"] = (LW - LW_grad*learning_rate, Lb - Lb_grad*learning_rate)

        return dL_dh

    def forward_step(self, x, c, h):
        f = self.linear_activation_forward(x, h, "f", "sigmoid")
        i = self.linear_activation_forward(x, h, "i", "sigmoid")
        g = self.linear_activation_forward(x, h, "g", "sigmoid")
        o = self.linear_activation_forward(x, h, "o", "tanh")

        c_new = f * c + i * g
        h_new = o * sigmoid(c_new)

        self.cache.append((h_new, (f, i, g, o, c_new, c)))
        
        return c_new, h_new

    def forward_linear(self, x):
        LW, Lb = self.params["L"]

        y = np.dot(LW, x) + Lb

        self.cache.append((LW, Lb, x, y))

        return y

    def backprop_linear(self, y):
        LW, Lb, h, y_out = self.cache.pop()

        dL_dy = square_loss_grad(y_out, y)

        # print(h)

        dL_dLW = np.dot(dL_dy, h.T)
        # print(dL_dLW)
        dL_db = np.sum(dL_dy, axis=1, keepdims=True)

        self.grads["L"] = (dL_dLW, dL_db)

        return np.dot(LW.T, dL_dy)

    def initialize_gradients(self):
        for param_name in param_names:
            W_grad = np.zeros((self.n_dims_hidden, self.n_dims_in))
            U_grad = np.zeros((self.n_dims_hidden, self.n_dims_hidden))
            b_grad = np.zeros((self.n_dims_hidden, 1))
            self.grads[param_name] = (W_grad, U_grad, b_grad)

        LW_grad = np.zeros((self.n_dims_out, self.n_dims_hidden))
        Lb_grad = np.zeros((self.n_dims_out, 1))
        self.grads["L"] = (LW_grad, Lb_grad)

    def forward(self, inp, h=None):
        raise NotImplementedError()
    
    def train_on_example(self, x, y, learning_rate):
        raise NotImplementedError()
            
class ManyToOneLSTM(BaseLSTM):
    def forward(self, inp, h=None):
        c = np.zeros((self.n_dims_hidden, 1))
        if not h:
            h = np.zeros((self.n_dims_hidden, 1))

        for x in inp:
            c, h = self.forward_step(x, c, h)

        y = self.forward_linear(h)

        return y

    def train_on_example(self, x, y, learning_rate):
        self.initialize_gradients()
        y_out = self.forward(x)

        dL_dh = self.backprop_linear(y)        
        self.backprop(dL_dh, learning_rate)

        return square_loss(y_out, y)


class OneToManyLSTM(BaseLSTM):
    def __init__(self, n_dims_in, n_dims_hidden, max_len):
        self.max_len = max_len
        super().__init__(n_dims_in, n_dims_hidden, n_dims_out=n_dims_in)
    def forward(self, inp, h=None):
        c = np.zeros((self.n_dims_hidden, 1))
        if not h:
            h = np.zeros((self.n_dims_hidden, 1))
        
        y = inp
        res = []

        for i in range(self.max_len):
            c, h = self.forward_step(y, c, h)
            y = self.forward_linear(h)
            res.append(y)

        return res

    def backprop_step(self, dL_dh,y):
        dL_dh += self.backprop_linear(y)
        return super().backprop_step(dL_dh)

    def train_on_example(self, x, y, learning_rate):
        self.initialize_gradients()
        y_out = self.forward(x)

        dL_dh = np.zeros((self.n_dims_hidden, 1))
        
        self.backprop(dL_dh, learning_rate, list(y))

        return np.sum(square_loss(y_out, y))
        