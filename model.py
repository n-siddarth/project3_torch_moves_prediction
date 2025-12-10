import csv

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from data_collection import BOARD_SIZE

with open("x_train.csv", "r") as f:
    X_train = torch.from_numpy(np.array(list(csv.reader(f)), dtype=np.float64))

with open("x_test.csv", "r") as f:
    X_test = torch.from_numpy(np.array(list(csv.reader(f)), dtype=np.float64))

with open("y_train.csv", "r") as f:
    y_train = torch.from_numpy(np.array(list(csv.reader(f)), dtype=np.float64))

with open("y_test.csv", "r") as f:
    y_test = torch.from_numpy(np.array(list(csv.reader(f)), dtype=np.float64))



class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.act = nn.ReLU()
        self.input = nn.Linear(BOARD_SIZE ** 2, 512, bias=True)
        self.h1 = nn.Linear(512, 512, bias=True)
        self.output = nn.Linear(512, 1, bias=True)
        self.double()


    def forward(self, x):
        x = self.input(x)
        x = self.act(x)
        x = self.h1(x)
        x = self.act(x)
        x = self.output(x)

        return x


if __name__ == "__main__":
    model = Model()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    # Training loop
    for _ in tqdm(range(epochs)):
        outputs = model(X_train)
        loss = loss_function(outputs, y_train)

        print("Current Loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing loop
        with torch.no_grad():
            outputs = model(X_test)
            test_loss = loss_function(outputs, y_test)

            print("Test Loss: ", test_loss.item())

    torch.save(model.state_dict(), "model.pt")
