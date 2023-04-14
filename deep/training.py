import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim
import torch.utils.data as data
from tool.create_dataset import creation

def training_lstm(model, device, dataset, num_epochs=200,batch=512,lr=0.0001,lookback=20):
    '''
    Training function for AirModel. Put in a different file for ease
    '''

    # dataset must be: ((X_train,y_train),(X_test,y_test))
    train, test = dataset
    X_train, y_train = train
    X_test, y_test = test
    

    # Training
    
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch)



    n_epochs = num_epochs
    for epoch in range(n_epochs):
        model.train()
        i=0
        for X_batch, y_batch in loader:
            i+=1
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            train_rmse = torch.sqrt(loss_fn(model(X_train), y_train))
            test_rmse = torch.sqrt(loss_fn(model(X_test), y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))