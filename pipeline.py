import torch
import deep.model as models
import deep.training as training
import pandas as pd
from tool.preprocessing import DataCollection
from tool.create_dataset import creation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import r2_score
from deep.training import training_mlp


#Config
PROB = 1
NUM_EPOCHS_ = 200
device= torch.device("cuda:0")

#Hyperparameters
NUM_RECORD = [6, 12, 20]
BATCH_SIZE=[64, 256, 512]
NUM_HIDDEN = [300, 500, 700]
NUM_LAYER = 6




#Data Loader
collection = DataCollection(drop_null=True)
gt = collection.get_gt()

X = torch.tensor([]).to(torch.device("cuda:0"))
y = torch.tensor([]).to(torch.device("cuda:0"))

import statistics

def plotter(model, tr_loss, ts_loss, r2test, folder,config):
    plt.clf()
    plt.title(config +f"\ntrain loss: {statistics.median(tr_loss[-7:]):.3f}\ntest loss: {statistics.median(ts_loss[-7:]):.3f}")
    plt.plot(tr_loss,'-g', label="Train loss,MSE")
    plt.plot(ts_loss,'-b', label="Test loss,MSE")
    leg = plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.savefig(folder+"loss.png")

    plt.clf()
    plt.title(config +f"\nr2 median value: {statistics.median(r2test[-7:])}")
    plt.plot(r2test,'-g', label="R2 test")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Value")
    leg = plt.legend(loc='lower right')
    plt.savefig(folder+"r2.png")

    plt.clf()

#modello
#device
import os
num_trains=1
res_trainings = list()
try:
    for record in NUM_RECORD:
        X = torch.tensor([]).to(torch.device("cuda:0"))
        y = torch.tensor([]).to(torch.device("cuda:0"))
        for i in collection.get_devices():
            tmp = pd.merge(i,gt,how="inner",on="valid_at").rename(columns={"pm2p5_y":"pm2p5_t","pm2p5_x":"pm2p5"})
            res = creation(tmp,lookback=record,p=1)
            X = torch.concat([X.clone(),res[0].flatten(-2)])
            y = torch.concat((y.clone(),res[1].flatten(-2)[:,0]))

        for batch_s in BATCH_SIZE:
            for hidden in NUM_HIDDEN:
                for i in range(5):
                    
                    if i==0:
                        model = models.AirMLP_6(num_fin=record*6, num_hidden=hidden).to(device)
                        model_name = "AirMLP_6"
                    if i==1:
                        model = models.AirMLP_7(num_fin=record*6, num_hidden=hidden).to(device)
                        model_name = "AirMLP_7"
                    if i==2:
                        model = models.AirMLP_8(num_fin=record*6, num_hidden=hidden).to(device)
                        model_name = "AirMLP_8"
                    if i==3:
                        model = models.AirMLP_7h(num_fin=record*6, num_hidden=hidden).to(device)
                        model_name = "AirMLP_7h"
                    if i==4:
                        model = models.AirMLP_8h(num_fin=record*6, num_hidden=hidden).to(device)
                        model_name = "AirMLP_8h"

                    config = f"record: {record} total_dim: {record*6}, batch_size: {batch_s}, hidden: {hidden}, model: {model}"
                    config_short = f"record: {record} total_dim: {record*6}, batch_size: {batch_s}, hidden: {hidden}, model: {model_name}"
                    print(f"Combination number: {num_trains}")
                    print(config)
                    res_tmp = training_mlp(X,y,model,batch_s,NUM_EPOCHS_,device)
                    
                    dir_new = rf"./results/trainings_{num_trains:03d}/"
                    if not os.path.exists(dir_new):
                        os.makedirs(dir_new)
                    num_trains+=1
                    plotter(model,res_tmp[0],res_tmp[1],res_tmp[2],dir_new,config_short)

                    #torch.save(model,dir_new+"weights.pth")
                    with open(dir_new+"config.txt","w") as f:
                        f.write(config)
                    with open(dir_new+"epoch_value.txt","w") as f:
                        f.write("=========================")
                        f.write(str(res_tmp[0]))
                        f.write("\n=========================")
                        f.write(str(res_tmp[1]))
                        f.write("\n=========================")
                        f.write(str(res_tmp[2]))
                        f.write("\n=========================")
        os.system(f"zip result_{num_trains}.zip ./results/")
        os.system(f"cp result_{num_trains}.zip ../drive/MyDrive/res_siws")
except KeyboardInterrupt:
    os.system(f"zip result_{num_trains}.zip ./results/")
    os.system(f"cp result_{num_trains}.zip ../drive/MyDrive/res_siws")