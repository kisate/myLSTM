from numpy.core.fromnumeric import size
from model import *
import numpy as np

n_dims_hidden = 100
emb_dims = 20
token_dims = 10
max_len = 10
n_samples = 1000
learning_rate = 1e-5

def to_one_hot(value):
    if value == -1:
        return np.zeros((token_dims, 1))
    return np.eye(token_dims)[value].reshape((token_dims, 1))

def from_one_hot(value):
    if np.linalg.norm(value) < eps:
        return -1
    return np.argmax(value)

params = {
        "n_dims_hidden" : n_dims_hidden,
        "enc_emb_dims" : emb_dims,
        "dec_emb_dims" : emb_dims,
        "loss_func" : "softmax_ce",
        "activation_func" : "softmax",
        "token_dims" : token_dims,
        "start_token" : to_one_hot(0),
        "max_len" : max_len
    }

eps = 1e-7

    

x = [np.random.randint(low=1, high=token_dims, size=np.random.randint(low=5, high=max_len-1)) for _ in range(n_samples)]
y = [[0] + sorted(val) + [-1]*(max_len - len(val) - 1) for val in x]

x_oh = [np.array([to_one_hot(val) for val in sample]) for sample in x]
y_oh = [np.array([to_one_hot(val) for val in sample]) for sample in y]

from tqdm import tqdm

model = DecoderWithEncoder(**params)


epochs = 25


for epoch in range(epochs):
    total_loss = 0
    for i in tqdm(range(n_samples)):
        model.initialize_gradients()
        y_out = model.forward(x_oh[i])
        model.backprop(list(y_oh[i]))
        model.update_parameters(learning_rate)
        total_loss += cross_entropy_loss(y_out, y_oh[i]).sum()

    print(f"Epoch {epoch}/{epochs}, loss : {total_loss / n_samples}")

x_test = np.random.randint(low=1, high=token_dims, size=np.random.randint(low=5, high=max_len-1))
x_test_oh = np.array([to_one_hot(val) for val in x_test])
model.enable_caching(False)
print(x)
y = model.forward(x_test_oh)
y = [from_one_hot(val) for val in y]
print(y)