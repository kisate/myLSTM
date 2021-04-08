from model import ManyToOneLSTM, OneToManyLSTM
import numpy as np


x_train = np.random.rand(500, 10, 1)
y_train = np.random.rand(500, 30, 10, 1)

model = OneToManyLSTM(10, 4, 30)


from tqdm import tqdm

def train(model, epochs, x_train, y_train, learning_rate):
    
    dataset = list(zip(x_train, y_train))
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataset):
            # print(x)
            total_loss += model.train_on_example(x, y, learning_rate)
        print (f"Epoch {epoch}/{epochs}: loss {total_loss / len(dataset)}")


train(model, 100, x_train, y_train, 0.001)