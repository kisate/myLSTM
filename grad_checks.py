import numpy as np
from functions import *
from model import *
from grad_checker import *

def check_square():
    n_dims = 10

    x = np.random.rand(n_dims, 1)
    y = np.random.rand(n_dims, 1)
    dL_dx = square_loss_grad(x, y)
    dL_dx_app = calc_grad(lambda x: square_loss(x, y), x)
    
    return dL_dx - dL_dx_app

def check_softmax_ce():
    n_dims = 10
    
    x = np.random.rand(n_dims, 1)
    y = np.zeros((n_dims, 1))
    y[np.random.randint(10)] = np.array([1])
    dL_dx = softmax_cross_entropy_loss_grad(x, y)
    dL_dx_app = calc_grad(lambda x: softmax_cross_entropy_loss(x, y), x)
    
    return dL_dx_app - dL_dx


def check_softmax():
    n_dims_in = 7
    n_dims_out = 7

    x = np.random.rand(n_dims_in, 1)
    y = np.random.rand(n_dims_out, 1)

    out = softmax(x)
    dL_dx = square_loss_grad(out, y)
    dL_dx = softmax_grad(dL_dx, x)
    
    dL_dx_app = calc_grad(lambda x: square_loss(softmax(x) , y), x).reshape(dL_dx.shape)
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
    dL_dx_app = calc_grad(lambda x: softmax_cross_entropy_loss(np.dot(W2, np.dot(W1, x) + b1) + b2 , y), x)
    return dL_dx_app - dL_dx


class OneToOneLSTM(LSTMWithOutput):
    def forward(self, inp, h=None, c=None):
        if c is None:
            c = np.zeros((self.n_dims_hidden, 1))
        if h is None:
            h = np.zeros((self.n_dims_hidden, 1))

        c, h = self.forward_step(inp, c, h)

        y, _ = self.linear_output(h)

        return y


def check_one_layer():
    n_dims_in = 7
    n_dims_hidden = 9
    n_dims_out = 4

    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "n_dims_out" : n_dims_out,
        "loss_func" : "softmax_ce",
        "output_activation" : "softmax"
    }

    model = OneToOneLSTM(**params)
    
    x = np.random.rand(n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden , 1)

    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    model.initialize_gradients()
    model.enable_caching = False

    dL_dh_app = calc_grad(lambda h: cross_entropy_loss(model.forward(x, h, c), y), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = model.backprop_output(y)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc)

    return dL_dh_app - dL_dh

def check_many_to_one():
    n_dims_in = 7
    n_dims_hidden = 9
    n_dims_out = 4
    n_samples = 2

    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "n_dims_out" : n_dims_out,
        "loss_func" : "square",
        "output_activation" : "id"
    }

    model = ManyToOneLSTM(**params)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden , 1)

    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    model.initialize_gradients()
    model.enable_caching = False

    dL_dh_app = calc_grad(lambda h: square_loss(model.forward(x, h, c), y), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = model.backprop_output(y)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc)

    return dL_dh_app - dL_dh

def check_many_to_many():
    n_dims_in = 10
    n_dims_hidden = 13
    n_dims_out = 7
    n_samples = 10


    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "n_dims_out" : n_dims_out,
        "loss_func" : "square",
        "output_activation" : "id"
    }

    model = ManyToManyLSTM(**params)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.zeros(h.shape)

    # y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])
    y = np.random.rand(n_samples, n_dims_out, 1)

    model.initialize_gradients()
    model.enable_caching = False

    # print(cross_entropy_loss(model.forward(x, h, c), y).sum())

    dL_dh_app = calc_grad(lambda h: square_loss(model.forward(x, h, c), y).sum(), h)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = np.zeros(h.shape)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, list(y))

    # print(dL_dh)
    # print(dL_dh_app)

    return dL_dh_app - dL_dh

def check_encoder():
    n_dims_in = 10
    n_dims_hidden = 13
    emb_dims = 7
    n_samples = 10

    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "embedding_dims" : emb_dims,
        "loss_func" : "square"
    }

    model = Encoder(**params)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.zeros(h.shape)

    # y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])
    y = np.random.rand(n_dims_hidden, 1)

    model.initialize_gradients()
    model.enable_caching = False

    # print(cross_entropy_loss(model.forward(x, h, c), y).sum())

    dL_dh_app = calc_grad(lambda h: square_loss(model.forward(x, h, c), y).sum(), h)

    model.enable_caching = True
    y_out = model.forward(x, h, c)

    dL_dh = square_loss_grad(y_out, y)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc)

    # print(dL_dh)
    # print(dL_dh_app)

    return dL_dh_app - dL_dh

def check_decoder():
    n_dims_hidden = 13
    emb_dims = 7
    n_dims_out = 10
    n_samples = 13

    y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])

    params = {
        "n_dims_hidden" : n_dims_hidden,
        "embedding_dims" : emb_dims,
        "loss_func" : "softmax_ce",
        "output_activation" : "softmax",
        "n_dims_out" : n_dims_out,
        "start_token" : y[0],
        "max_len" : n_samples
    }

    model = Decoder(**params)

    x = params["start_token"]
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden, 1)

    model.initialize_gradients()
    model.enable_caching = False

    dL_dh_app = calc_grad(lambda h: cross_entropy_loss(model.forward(x, h, c), y).sum(), h)

    model.enable_caching = True
    y_out = model.forward(x, h, c)

    dL_dh = np.zeros(h.shape)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros((n_dims_out, 1))
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, list(y))

    return dL_dh_app - dL_dh

def check_decoder_with_encoder():
    n_dims_hidden = 13
    emb_dims = 7
    token_dims = 10
    n_samples = 13

    y = np.array([np.array([x]).T for x in np.eye(token_dims)[np.random.choice(token_dims, n_samples)]])
    x = np.array([np.array([x]).T for x in np.eye(token_dims)[np.random.choice(token_dims, n_samples)]]) 
    params = {
        "n_dims_hidden" : n_dims_hidden,
        "enc_emb_dims" : emb_dims,
        "dec_emb_dims" : emb_dims,
        "loss_func" : "softmax_ce",
        "activation_func" : "softmax",
        "token_dims" : token_dims,
        "start_token" : y[0],
        "max_len" : n_samples
    }

    model = DecoderWithEncoder(**params)

    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden, 1)

    model.initialize_gradients()
    model.enable_caching(False)

    dL_dh_app = calc_grad(lambda h: cross_entropy_loss(model.forward(x, h, c), y).sum(), h)

    model.enable_caching = True
    y_out = model.forward(x, h, c)

    dL_dh = np.zeros(h.shape)
    dL_dc = np.zeros(h.shape)

    dL_dh, dL_dc = model.backprop(list(y))

    return dL_dh_app - dL_dh

def check_decoder_params():
    # n_dims_in = 10
    n_dims_hidden = 13
    emb_dims = 7
    n_dims_out = 10
    n_samples = 13

    y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])

    params = {
        "n_dims_hidden" : n_dims_hidden,
        "embedding_dims" : emb_dims,
        "loss_func" : "square",
        "output_activation" : "id",
        "n_dims_out" : n_dims_out,
        "start_token" : y[0],
        "max_len" : n_samples
    }

    model = Decoder(**params)

    x = params["start_token"]
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden, 1)

    y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, params["max_len"])]])
    # print(y.shape)
    # y = np.random.rand(params["max_len"], n_dims_out, 1)
    model.initialize_gradients()
    model.enable_caching = False

    results = calc_param_grads(lambda model: square_loss(model.forward(x, h, c), y).sum(), model)

    model.enable_caching = True
    y_out = model.forward(x, h, c)

    dL_dh = np.zeros(h.shape)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros((n_dims_out, 1))
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, list(y))
    
    return collect_param_results(model.grads.items(), results)

def check_one_to_one_params():
    n_dims_in = 7
    n_dims_hidden = 9
    n_dims_out = 4

    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "n_dims_out" : n_dims_out,
        "loss_func" : "softmax_ce",
        "output_activation" : "softmax"
    }

    model = OneToOneLSTM(**params)
    
    x = np.random.rand(n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.random.rand(n_dims_hidden , 1)

    y = np.zeros((n_dims_out, 1))
    y[np.random.randint(n_dims_out)] = np.array([1])

    model.initialize_gradients()
    model.enable_caching = False

    results = calc_param_grads(lambda model: cross_entropy_loss(model.forward(x, h, c), y), model)

    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = model.backprop_output(y)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc)

    return collect_param_results(model.grads.items(), results)

def check_many_to_many_params():
    n_dims_in = 10
    n_dims_hidden = 13
    n_dims_out = 7
    n_samples = 2

    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "n_dims_out" : n_dims_out,
        "loss_func" : "square",
        "output_activation" : "id"
    }

    model = ManyToManyLSTM(**params)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.zeros(h.shape)

    # y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])
    y = np.random.rand(n_samples, n_dims_out, 1)

    model.initialize_gradients()
    model.enable_caching = False

    # print(cross_entropy_loss(model.forward(x, h, c), y).sum())
    results = calc_param_grads(lambda model: square_loss(model.forward(x, h, c), y).sum(), model)


    model.enable_caching = True
    model.forward(x, h, c)

    dL_dh = np.zeros(h.shape)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc, list(y))

    return collect_param_results(model.grads.items(), results)

def check_encoder_params():
    n_dims_in = 10
    n_dims_hidden = 13
    emb_dims = 10
    n_samples = 10

    params = {
        "n_dims_in" : n_dims_in,
        "n_dims_hidden" : n_dims_hidden,
        "embedding_dims" : emb_dims,
        "loss_func" : "square"
    }

    model = Encoder(**params)

    x = np.random.rand(n_samples, n_dims_in, 1)
    h = np.random.rand(n_dims_hidden, 1)
    c = np.zeros(h.shape)

    # y = np.array([np.array([x]).T for x in np.eye(n_dims_out)[np.random.choice(n_dims_out, n_samples)]])
    y = np.random.rand(n_dims_hidden, 1)

    model.initialize_gradients()
    model.enable_caching = False

    # print(cross_entropy_loss(model.forward(x, h, c), y).sum())

    results = calc_param_grads(lambda model: square_loss(model.forward(x, h, c), y).sum(), model)

    model.enable_caching = True
    y_out = model.forward(x, h, c)

    dL_dh = square_loss_grad(y_out, y)
    dL_dc = np.zeros(h.shape)
    dL_dx = np.zeros(x.shape)
    dL_dh, dL_dc = model.backprop(dL_dh, dL_dc)

    return collect_param_results(model.grads.items(), results)

