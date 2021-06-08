import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision  
import torchvision.transforms as transforms
from datetime import datetime
import pandas as pd
import os
from os.path import join
import timeit
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import optuna
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance



# Here I load auxiliar functions and the "define_Generator" and "define_Discriminator"
from aux import *
from architecture import *



class Engine:
    def __init__(self):
        super().__init__()

    ########################################################################
    #                           AUX FUNCTIONS                              #
    ########################################################################

    def _load_data(self, load_train=True, load_val=True):
        # Load train data
        if load_train:
            self.data = data_loader(data='train')

        # Load validation data
        if load_val:
            val_data = data_loader(data='validate')

            # Validation data prep.
            _, self.val_features, self.val_label, self.val_weights, _ = next(create_batch(val_data, size=val_data.shape[0], device='cpu'))
            del val_data # Saves space

            # Puting the features of validation data to gpu to pass it through the D later
            self.val_features = self.val_features.to(device)

    def _define_batch_loader(self, data_type, size, device):
        batch_loader = create_batch(self.data, size=size, debug=False, device=device)
        data_shape = self.data.shape 
        return batch_loader, data_shape

    ########################################################################
    #                             TRAIN D/G                                #
    ########################################################################

    def train_discriminator(self, real, fake, weights, BATCH_SIZE):
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))

        # Zerar gradientes
        self.disc.zero_grad()

        # Passar oos dados *reais* (SD) pelo discriminator e calcular a loss
        disc_real = self.disc(real).reshape(-1)
        loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real)*0.1) # Real (background) == 0, usamos o *0.1 de modo a nao estar demasiado confiante

        # Passar o *fake* pelo discriminator e calcular a loss
        disc_fake = self.disc(fake.detach()).reshape(-1) # O detach é para nao treinar os gradientes do generator
        loss_disc_fake = self.criterion(disc_fake, torch.ones_like(disc_fake)*0.9) #  Sinal == 1, usamos o *0.9 de modo a nao estar demasiado confiante

        # Loss do discriminator
        loss_disc = loss_disc_real + loss_disc_fake

        # Weights
        # Multiplicar os pesos pela batch size, a sua soma será igual a ela.
        weights = weights * BATCH_SIZE
        loss_disc = (weights * loss_disc) / weights.sum()
        loss_disc = torch.mean(loss_disc, dtype=torch.float32)
        loss_disc.backward() # Calculate gradients
        self.opt_disc.step() # Update weights of the network

        return loss_disc.item()
        


    def train_generator(self, fake, weights):
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        
        # Zerar gradientes
        self.gen.zero_grad()

        # Passar o fake pelo discriminator
        output = self.disc(fake).reshape(-1)

        # Calculate the loss of G
        loss_gen = self.criterion(output, torch.ones_like(output)*0.9)

        # Weights
        # Set all the weights to one, a sua soma será igual à batch_size
        weights = torch.ones(weights.shape).to(device)
        loss_gen = (weights * loss_gen) / weights.sum()
        loss_gen = torch.mean(loss_gen, dtype=torch.float32)
        loss_gen.backward() # Calculate gradients
        self.opt_gen.step()# Update weights of the network

        return loss_gen.item()

    
    ########################################################################
    #                            TRAIN LOOP                                #
    ########################################################################


    def train(self
            , trial
            , FEATURES = 68
            , name = ""):

        print("[+] Inicializing training..\n")
    
        ###############################
        #    OPTUNA & Define Models   #
        ###############################
        NUM_EPOCHS = trial.suggest_int("n_epochs", 1, 1000)
        BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 100, 500)
        NOISE_DIM = trial.suggest_int("NOISE_DIM", 100, 300)
        early_stoping = trial.suggest_int("early_stoping", 3, 1000)

        # Define Models
        self.disc = define_Discriminator(trial, FEATURES).to(device)
        self.gen = define_Generator(trial, NOISE_DIM, FEATURES).to(device)

        # Set them to train
        self.disc = self.disc.train()
        self.gen = self.gen.train()
        
        # Criterion
        self.criterion = nn.BCELoss()
        
        ## Define optimizers
        # TODO: Betas are hardcoded
        # G
        optimizer_name_gen = trial.suggest_categorical("optimizer_gen", ["SGD", "RMSprop", "Adam"])
        lr_gen = trial.suggest_float("lr_gen", 1e-5, 1e-1, log=True)
        if optimizer_name_gen != "Adam":
            self.opt_gen = getattr(optim, optimizer_name_gen)(self.gen.parameters(), lr=lr_gen)
        else:
            self.opt_gen = getattr(optim, optimizer_name_gen)(self.gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        # D
        optimizer_name_disc = trial.suggest_categorical("optimizer_disc", ["SGD", "RMSprop", "Adam"])
        lr_disc = trial.suggest_float("lr_disc", 1e-5, 1e-1, log=True)
        if optimizer_name_disc != "Adam":
            self.opt_disc = getattr(optim, optimizer_name_disc)(self.disc.parameters(), lr=lr_disc)
        else:
            self.opt_disc = getattr(optim, optimizer_name_disc)(self.disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        ###############################

        ## Define variables I want to keep track (Tensorboard)
        variables = ["loss_D", "loss_G", "wasserstein_distance", "roc_score", "best_wd", "best_roc"]
        
        self.supervisor = Supervisor(variables, name)

        # Set them all to 0
        self.supervisor.define(variables, 0)

        # Other variables
        best_roc = None
        best_wd = None
        patience= 0
         
        for epoch in range(NUM_EPOCHS):

            # Put D to train mode
            self.disc = self.disc.train()

            # Define batch_loader
            batch_loader, data_shape = self._define_batch_loader(data_type="train", size=BATCH_SIZE, device=device)

            ###############################
            #        TRAINING LOOP        #
            ###############################
            for _, real, label, weights, _ in tqdm(batch_loader, total=data_shape[0]/BATCH_SIZE):
                # Real shape - [batch_size, 68]
                # Label shape - [batch_size, 1]
                # Weights shape - [batch_size]

                # Define sobre batch variables
                noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
                fake = self.gen(noise) # This  is the fake data | shape = [100, 68]

                # Train
                loss_D = self.train_discriminator(real, fake, weights, BATCH_SIZE)

                loss_G = self.train_generator(fake, weights)


            # Report losses
            self.supervisor.report("loss_D", loss_D, epoch)
            self.supervisor.report("loss_G", loss_G, epoch)


            ##########################################################
            ## Calculate & Report Wasserstein Distance
            wd_score = 0
            for x in range(real.shape[0]):
                wd_score += wasserstein_distance(
                                    real[x].cpu().reshape(-1).tolist() # Tem 68
                                , fake[x].cpu().reshape(-1).tolist() # Tem 68
                                , (weights[x].cpu() * BATCH_SIZE * torch.ones([68])).tolist()
                                , torch.ones([68]).tolist()
                                )
            wd_score = wd_score/BATCH_SIZE

            # Report WD Score
            self.supervisor.report("wasserstein_distance", wd_score, epoch)


            ## Calculate & Report Roc Score
            self.disc = self.disc.eval()
            with torch.no_grad():

                roc_score = roc_auc_score(self.val_label, (self.disc(self.val_features).squeeze(1)).cpu(), sample_weight=self.val_weights).item()
                
                # Report ROC Score
                self.supervisor.report("roc_score", roc_score, epoch)

            ## Save the weights
            if best_roc is None:
                best_roc = roc_score
            if best_wd is None:
                best_wd = wd_score

            # Trying to find max
            if roc_score > best_roc:
                print(f"[!] Best ROC Yet ({roc_score}), saving...")
                best_roc = roc_score

                # Save D
                torch.save(self.disc.state_dict(), join(os.getcwd(), "models",f"{name}_disc.plk"))

            # Trying to find min
            if wd_score < best_wd:
                best_wd = wd_score
                print(f"[!] Best WD Yet ({wd_score}), saving...")
                # Save G
                torch.save(self.gen.state_dict(), join(os.getcwd(), "models",f"{name}_gen.plk"))

            # Report best scores
            self.supervisor.report("best_roc", best_roc, epoch)
            self.supervisor.report("best_wd", best_wd, epoch)

            # If there is some improvement on the roc OR wd it will reset patience
            if wd_score == best_wd or best_roc == roc_score:
                patience= 0
            else:
                patience +=1
            if early_stoping == patience:
                print("[-] Early stoping brake!")
                break

                

            print(
                f"[+] Epoch {epoch} completed | LossG: {round(loss_G, 10)} | LossD: {round(loss_D, 10)} | WD: {round(wd_score, 10)} | ROC: {round(roc_score, 10)} | Best ROC: {round(best_roc, 10)} | Best WD: {round(best_wd, 10)} | Early Stopping {patience}/{early_stoping}"
                )


        self.supervisor.flush()

        #return best_score


    ########################################################################
    #                        OPTUNA OPTIMIZATION                           #
    ########################################################################

    def optuna_optimization(self, study, n_trials=None, timeout=None):

        # Se o estudo já existir vai la buscar o global_score se não é 0
        try:
            self.global_score = study.best_trial.value
            print("Global_Score loaded from previous study!", self.global_score)
        except:
            self.global_score = 0

        study.optimize(self.train, n_trials=n_trials, timeout=timeout)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
