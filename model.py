from functions import *

import numpy as np

param_names = [
    "f", "i", "g", "o"
]


activation_functions = {
    "sigmoid" : (sigmoid, sigmoid_grad),
    "tanh" : (tanh, tanh_grad),
    "softmax" : (softmax, softmax_grad),
    "id" : (lambda x : x, lambda x, _: x)
}

loss_functions = {
    "square" : (square_loss, square_loss_grad),
    "softmax_ce" : (softmax_cross_entropy_loss, softmax_cross_entropy_loss_grad)
}

class BaseLSTM:
    """Base class for LSTM.
    
    forward() should be implemented in child 

    forward() with enabled caching should be ran before backprop()

    params:
    n_dims_in, n_dims_out, loss_func

    """
    def __init__(self, params):
        self.n_dims_in = params["n_dims_in"]
        self.n_dims_hidden = params["n_dims_hidden"]
        self.loss_func = params["loss_func"]
        
        self.params = {}
        self.grads = {}
        
        for param_name in param_names:
            W = np.random.rand(self.n_dims_hidden, self.n_dims_in) * 0.01
            U = np.random.rand(self.n_dims_hidden, self.n_dims_hidden) * 0.01
            b = np.random.rand(self.n_dims_hidden, 1) * 0.01
            self.params[param_name] = (W, U, b)
        
        self.cache = []
        self.enable_caching = True

    def linear_activation_forward(self, x, h, param_name, func_name):
        '''Linear layer with activation. Modifies cache'''
        W, U, b = self.params[param_name]
        activation_function = activation_functions[func_name][0]

        z = np.dot(W, x) + np.dot(U, h) + b
        a = activation_function(z)

        self.save_to_cache((param_name, x, h, z, a, func_name))
        return a

    def backprop_step_linear(self, dL_da):
        '''Backprop linear gate with activation. Modifies cache'''
        param_name, x, h, z, a, func_name = self.cache.pop()
        W, U, b = self.params[param_name]
        activation_function_grad = activation_functions[func_name][1]

        dL_dz = activation_function_grad(dL_da, z)
        
        dL_dW = np.dot(dL_dz, x.T)
        
        dL_dU = np.dot(dL_dz, h.T)
        dL_dh = np.dot(U.T, dL_dz)

        dL_dx = np.dot(W.T, dL_dz)

        dL_db = np.sum(dL_dz, axis=1, keepdims=True) 

        W_grad, U_grad, b_grad = self.grads[param_name]
        
        self.grads[param_name] = (W_grad + dL_dW, U_grad + dL_dU, b_grad + dL_db)

        return dL_dh, dL_dx

    def backprop_step_no_output(self, dL_dh, dL_dc, cache):
        '''Backprop hidden layer. Modifies cache'''
        f, i, g, o, c_new, c = cache

        dL_do = tanh(c_new)*dL_dh

        dL_dc = dL_dc + tanh_grad(dL_dh * o, c_new)
        
        dL_df = dL_dc * c
        dL_di = dL_dc * g 
        dL_dg = dL_dc * i

        do_dh, do_dx = self.backprop_step_linear(dL_do)
        dg_dh, dg_dx = self.backprop_step_linear(dL_dg)
        di_dh, di_dx = self.backprop_step_linear(dL_di)
        df_dh, df_dx = self.backprop_step_linear(dL_df)

        dL_dh = do_dh + df_dh + dg_dh + di_dh 
        dL_dc = dL_dc*f
        dL_dx = do_dx + df_dx + dg_dx + di_dx

        return dL_dh, dL_dc, dL_dx

    def backprop_step(self, dL_dh, dL_dc, data=None):
        '''Backpropagation step'''
        h, cache = self.cache.pop()
        return self.backprop_step_no_output(dL_dh, dL_dc, cache)

    
    def backprop(self, dL_dh, dL_dc, data_stack=None):
        '''Clear cache and calculate gradients'''
        while self.cache:
            if data_stack is None:
                dL_dh, dL_dc, dL_dx = self.backprop_step(dL_dh, dL_dc)
            else:
                dL_dh, dL_dc, dL_dx = self.backprop_step(dL_dh, dL_dc, data_stack.pop())
                
        return dL_dh, dL_dc

    def update_params(self, learning_rate):
        for param_name, param in self.params.items():
            grads = self.grads[param_name]
            self.params[param_name] = tuple(param[i] - grads[i]*learning_rate for i in range(len(grads)))

    def forward_step(self, x, c, h):
        '''Hidden layer forward'''
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
        for param_name, params in self.params.items():
            self.grads[param_name] = tuple(np.zeros(param.shape) for param in params)

    def save_to_cache(self, value):
        if (self.enable_caching):
            self.cache.append(value)

    def forward(self, inp, h=None, c=None):
        raise NotImplementedError()
    

class LSTMWithOutput(BaseLSTM):
    """
    Base class for LSTMs with output activation function. 

    forward is not implemented.

    loss_func takes output before running through activation

    For example: loss_func = 'sofmtax_ce', output_activation = 'softmax'

    params:
    n_dims_out, n_dims_in, n_dims_hidden, loss_func, output_activation

    """
    def __init__(self, params):
        super().__init__(params)
        self.n_dims_out = params["n_dims_out"]
        self.output_activation = params["output_activation"]
        
        LW = np.random.rand(params["n_dims_out"], params["n_dims_hidden"]) * 0.01
        Lb = np.random.rand(params["n_dims_out"], 1) * 0.01

        self.params["Lo"] = (LW, Lb)

    def linear_output(self, x):
        '''Apply activation function to output'''
        LW, Lb = self.params["Lo"]

        y = np.dot(LW, x) + Lb

        self.save_to_cache((LW, Lb, x, y))

        return activation_functions[self.output_activation][0](y), x

    def backprop_output(self, y):
        '''Calculate gradient for hidden vector'''
        LW, Lb, h, y_out = self.cache.pop()

        dL_dy = loss_functions[self.loss_func][1](y_out, y)

        dL_dLW = np.dot(dL_dy, h.T)

        dL_db = np.sum(dL_dy, axis=1, keepdims=True)

        W_grad, b_grad = self.grads["Lo"]
        
        self.grads["Lo"] = (W_grad + dL_dLW, b_grad + dL_db)

        return np.dot(LW.T, dL_dy)

class ManyToOneLSTM(LSTMWithOutput):
    def forward(self, inp, h=None, c=None):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
            
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))

        for x in inp:
            c, h = self.forward_step(x, c, h)

        y, _ = self.linear_output(h)

        return y

class ManyToManyLSTM(LSTMWithOutput):
    """
    params:
    n_dims_in, n_dims_hidden, loss_func, n_dims_out, output_activation
    """
    def __init__(self, params):
        super().__init__(params)

    def forward(self, inp, h=None, c=None):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))
        
        res = []

        for x in inp:
            c, h = self.forward_step(x, c, h)
            y, _ = self.linear_output(h)
            res.append(y)

        return np.array(res)

    def backprop_step(self, dL_dh, dL_dc, data):
        return super().backprop_step(dL_dh + self.backprop_output(data), dL_dc)

class LSTMWithEmbeddings(BaseLSTM):
    """
    params:
    n_dims_in, n_dims_hidden, loss_func, embedding_dims
    """
    def __init__(self, params):
        n_dims_in = params["n_dims_in"]
        params["n_dims_in"] = params["embedding_dims"]
        super().__init__(params)
        params["n_dims_in"] = n_dims_in
        self.n_dims_in = n_dims_in

        self.embedding_dims = params["embedding_dims"]

        LW = np.random.rand(params["embedding_dims"], params["n_dims_in"]) * 0.01
        Lb = np.random.rand(params["embedding_dims"], 1) * 0.01
        self.params["Le"] = (LW, Lb)
    
    def linear_embedding(self, x):
        '''Calculate linear embedding'''
        LW, Lb = self.params["Le"]

        emb = np.dot(LW, x) + Lb

        self.save_to_cache((LW, Lb, x, emb))

        return emb

    def backprop_embedding(self, dL_dx):
        '''Calculate gradient for embeddings'''
        LW, _, h, _ = self.cache.pop()

        dL_dLW = np.dot(dL_dx, h.T)

        dL_db = np.sum(dL_dx, axis=1, keepdims=True)

        W_grad, b_grad = self.grads["Le"]
        
        self.grads["Le"] = (W_grad + dL_dLW, b_grad + dL_db)

        return np.dot(LW.T, dL_dx)

    def backprop_step(self, dL_dh, dL_dc, data=None):
        dL_dh, dL_dc, dL_dx = super().backprop_step(dL_dh, dL_dc) 
        self.backprop_embedding(dL_dx)
        return dL_dh, dL_dc, dL_dx

class Encoder(LSTMWithEmbeddings):
    def forward(self, inp, h, c):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))    

        for x in inp:
            c, h = self.forward_step(self.linear_embedding(x), c, h)

        return h
        

class Decoder(LSTMWithEmbeddings, LSTMWithOutput):
    """
    params:
    n_dims_hidden, loss_func, embedding_dims, n_dims_out, output_activation, start_token, max_len
    """
    def __init__(self, params):
        params["n_dims_in"] = params["n_dims_out"]
        super().__init__(params)
        self.start_token = params["start_token"]
        self.max_len = params["max_len"]
        self.dL_dx = np.zeros((self.n_dims_out, 1))
    
    def forward(self, inp, h, c):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))    

        res = [self.start_token]

        while len(res) < self.max_len:
            e = self.linear_embedding(res[-1])
            
            c, h = self.forward_step(e, c, h)
            y, _ = self.linear_output(h)
            
            # token = np.zeros(y.shape)
            # token[y.argmax()] = 1
            res.append(y)

        return np.array(res)

    def backprop_step(self, dL_dh, dL_dc, data):
        dL_dx = self.dL_dx
        dL_dh = dL_dh + self.backprop_output(data, dL_dx)
        _, cache = self.cache.pop()
        
        dL_dh, dL_dc, dL_dx_t = self.backprop_step_no_output(dL_dh, dL_dc, cache) 
        
        dL_dx = self.backprop_embedding(dL_dx_t)
        self.dL_dx = dL_dx
        return dL_dh, dL_dc, dL_dx

    def backprop_output(self, y, dL_dy):
        '''Calculate gradient for hidden vector'''
        LW, Lb, h, y_out = self.cache.pop()

        dL_dy = loss_functions[self.loss_func][1](y_out, y) + activation_functions[self.output_activation][1](dL_dy, y_out)

        dL_dLW = np.dot(dL_dy, h.T)

        dL_db = np.sum(dL_dy, axis=1, keepdims=True)

        W_grad, b_grad = self.grads["Lo"]
        
        self.grads["Lo"] = (W_grad + dL_dLW, b_grad + dL_db)

        return np.dot(LW.T, dL_dy)

    # def propagate_from_dx(self, dL_dx):

