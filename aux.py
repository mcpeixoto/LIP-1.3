import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision  
import torchvision.transforms as transforms
from datetime import datetime
import pandas as pd
import os
import random
from os.path import join
import timeit
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import optuna
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
import shutil
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

print("[+] Defined all python modules with success")

# Data path
data_path = "/mnt/D/estagio_lip/estagio_lip_2/NormalizedData.h5"

# Define pyTorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print("[-] Unfortunally pytorch couldn't load GPU, CPU loaded instead.")
else:
    print("[+] Pytorch device:", device)


def data_loader(data="train", directory=data_path):
    # Data types = train, validate, test
    print(f"[+] Loading {data} data..")
    tic=timeit.default_timer()

    df = pd.read_hdf(
                    directory,
                    key=data,
                    )
    

    toc=timeit.default_timer()

    if data == "train":
        print("\t - Since training data was selected, only background will be loaded. [HARDCODED]")
        df = df[df['Label']==0] # So quero o bkg neste caso

    print("\t - Compleatly loaded in", int(toc-tic), "seconds!")

    return df


def create_batch(df, size=200, debug=False, device=device):
    # Data types = train, validate, test

    if debug:
        print("[Info] Len(data) =", len(df))
        print("[Info] Len(data) / Batch Size =", len(df)/size)

    

    # Select the data size
    # TODO: Estou a perder um pouco de dados no final
    batch = iter([
            # Big tuple
            (
                i+1, # Batch number
                torch.tensor(df.iloc[x:x+size].drop(columns=['index','Name', 'Weights', 'Label']).values, dtype=torch.float32).to(device), # Features Removi MissingET_Eta porque tava cheio de NaA (???)
                torch.tensor(df['Label'].iloc[x:x+size].values, dtype=torch.float32).to(device),

                torch.tensor((
                np.where(
                    df['Label'].iloc[x:x+size] == 0,
                    df['Weights'].iloc[x:x+size] / df['Weights'].iloc[x:x+size][df['Label'].iloc[x:x+size] == 0].sum(),
                    df['Weights'].iloc[x:x+size] / df['Weights'].iloc[x:x+size][df['Label'].iloc[x:x+size] == 1].sum(),
                )
                * df['Label'].iloc[x:x+size].shape[0]
                / 2
                ), dtype=torch.float32).to(device),
                df.iloc[x:x+size]['Name'] #Nome
        ) 
        
         for i,x in enumerate(list(filter(lambda x: (x%(size+1) == 0) , [x for x in range(len(df))])))]) # Para size = 100 -> 0, 101, 202, 303, .. , 909, 1010, 1111


    del df

    return batch



class Supervisor:
    """
    This class will keep record of a list of variables:
    update them, retrieve them and log it on tensorboard.
    """
    def __init__(self, variables, name, tensorboard_dir="logs", cwd=os.getcwd()):
        
        # Defining "global" variables
        self.cwd = cwd
        self.tensorboard_dir = tensorboard_dir

        
        if type(variables) != list:
            variables = [variables]
        
        # Record
        self.record = {x:None for x in variables}

        # Tensorboard Writer
        self.writer = SummaryWriter(join(cwd, tensorboard_dir, name))

    def _clean_log(self):
        shutil.rmtree(join(self.cwd, self.tensorboard_dir))

    def _plusOne(self, var):
        self.record[var] = self.record[var] + 1
        return self.record[var] 

    def define(self, variables, value):
        if type(variables) != list:
            variables = [variables]

        for x in variables:
            # Update record
            self.record[x] = value 

    def flush(self):
        self.writer.flush()


    def report(self, var, value, epoch):
        """
        This will save a variable to the record
        and write it to tensorboard
        TODO: https://pytorch.org/docs/stable/tensorboard.html
        """
        # Update record
        self.record[var] = value 

        # Update writer
        self.writer.add_scalar(var, value, epoch)
        
    def retrieve(self, var, if_None=0):
        """
        It returns the value of a variable in record.
        """
        value = self.record[var]

        if value == None:
            return if_None
        else:
            return value




print("\n \U0001f600 Happy codding! \U0001f600")