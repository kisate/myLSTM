from model import ManyToOneLSTM, OneToManyLSTM, ManyToManyLSTM
import numpy as np
from functions import softmax


n_classes = 20
n_samples = 10

x_train = np.random.rand(n_samples, 1, n_classes, 1)
# y_train = np.eye(n_classes)[np.random.choice(n_classes, n_samples)]
y_train = np.array([np.array([np.array([x]).T for x in np.eye(n_classes)[np.random.choice(n_classes, 1)]]) for _ in range(n_samples)])

print(x_train.shape)
print(y_train.shape)

# print(y_train)

model = ManyToManyLSTM(n_classes, 7, "square", softmax, n_classes)

losses = []


from tqdm import tqdm

def train(model, epochs, x_train, y_train, learning_rate):
    
    dataset = list(zip(x_train, y_train))
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataset):
            # print(x)
            total_loss += model.train_on_example(x, y, learning_rate)
        losses.append(total_loss/len(dataset))
        print (f"Epoch {epoch}/{epochs}: loss {total_loss / len(dataset)}")


from functions import softmax_cross_entropy_loss

y = y_train[0][0]
y_out = np.random.rand(n_classes, 1)

print(softmax_cross_entropy_loss(y_out, y))

train(model, 1, x_train, y_train, 0.001)

# import matplotlib.pyplot as plt

# print(losses)

# plt.plot(losses)
# plt.show()