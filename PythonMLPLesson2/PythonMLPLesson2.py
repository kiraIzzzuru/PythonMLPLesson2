import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torchvision as tv
import time

BATCH_SIZE = 256

train_dataset = tv.datasets.FashionMNIST('.', train=True, transform=tv.transforms.ToTensor(), download=True)
test_dataset = tv.datasets.FashionMNIST('.', train=False, transform=tv.transforms.ToTensor(), download=True)
train = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE)
test = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE)

print(train_dataset[0][0].shape)

# Flatten - Вытягивает квадратную матрицу в плоский вектор
# Linear - сжатие
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 512),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(0.5),
    # torch.nn.Linear(2560, 1280),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(0.5),
    # torch.nn.Linear(1280, 512),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(512),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(256),
    # torch.nn.Linear(256, 128),
    # torch.nn.ReLU(),
    # torch.nn.BatchNorm1d(128),
    # torch.nn.Linear(128, 64),
    # torch.nn.ReLU(),
    # torch.nn.BatchNorm1d(64),
    # torch.nn.Linear(64, 32),
    # torch.nn.ReLU(),
    # torch.nn.BatchNorm1d(32),
    torch.nn.Linear(256, 10)
    )

# функция потерь
loss = torch.nn.CrossEntropyLoss()
trainer = torch.optim.Adagrad(model.parameters(), lr=.01)
num_epoch = 10

def train_model():
    for ep in range(num_epoch):
        train_iters, train_passed = 0, 0
        train_loss, train_acc = 0., 0.
        start = time.time()
        
        model.train()
        for X, y in train:
            trainer.zero_grad()
            y_pred = model(X)
            l = loss(y_pred, y)
            l.backward()
            trainer.step()
            train_loss += l.item()
            train_acc += (y_pred.argmax(dim=1) == y).sum().item()
            train_iters += 1
            train_passed += len(X)
            
        test_iters, test_passed = 0, 0
        test_loss, test_acc = 0., 0.
        # переключаем режим для работы с тестовыми данными
        model.eval()
        for X, y in test:
            y_pred = model(X)
            l = loss(y_pred, y)
            test_loss += l.item()
            test_acc += (y_pred.argmax(dim=1) == y).sum().item()
            test_iters += 1
            test_passed += len(X)
            
        print("ep: {}, taked: {:.3f}, train_loss: {}, train_acc: {}, test_loss: {}, test_acc: {}".format(
            ep, time.time() - start, train_loss/train_iters, 
            train_acc/train_passed, test_loss/test_iters, test_acc/test_passed))
        
train_model()