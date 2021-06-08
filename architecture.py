import torch
import torch.nn as nn
import torch.nn.functional as F


def define_Discriminator(trial, in_features=68, _out_features=1):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers_disc", 1, 10)
    layers = []

    activation_name = trial.suggest_categorical("activation_disc", ["ReLU", "LeakyReLU", "Tanh", "PReLU"])

    for i in range(n_layers):
        out_features = trial.suggest_int("disc_n_units_l{}".format(i), 5, 264)
        layers.append(nn.Linear(in_features, out_features))

        # Activation layer
        if activation_name == "ReLU":
            layers.append(nn.ReLU())
        elif activation_name == "LeakyReLU":
            layers.append(nn.LeakyReLU())
        elif activation_name == "Tanh":
            layers.append(nn.Tanh())
        elif activation_name == "PReLU":
            layers.append(nn.PReLU())

        p = trial.suggest_float("disc_dropout_disc_l{}".format(i), 0.1, 0.7)
        layers.append(nn.Dropout(p))

        in_features = out_features

    # Ultima layer
    layers.append(nn.Linear(in_features, _out_features))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

def define_Generator(trial, in_features=100, _out_features=68):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers_gen", 1, 10)
    layers = []
    activation_name = trial.suggest_categorical("activation_gen", ["ReLU", "LeakyReLU", "Tanh", "PReLU"])

    for i in range(n_layers):
        out_features = trial.suggest_int("gen_n_units_l{}".format(i), 5, 264)
        layers.append(nn.Linear(in_features, out_features, bias=False))

        # Activation layer
        if activation_name == "ReLU":
            layers.append(nn.ReLU())
        elif activation_name == "LeakyReLU":
            layers.append(nn.LeakyReLU())
        elif activation_name == "Tanh":
            layers.append(nn.Tanh())
        elif activation_name == "PReLU":
            layers.append(nn.PReLU())

        in_features = out_features

    # Ultima layer
    layers.append(nn.Linear(in_features, _out_features, bias=False))

    last_activation = trial.suggest_categorical("gen_activation_last", ["_", "ReLU", "LeakyReLU", "Tanh", "PReLU"])

    if last_activation == "ReLU":
        layers.append(nn.ReLU())
    elif last_activation == "LeakyReLU":
        layers.append(nn.LeakyReLU())
    elif last_activation == "Tanh":
        layers.append(nn.Tanh())
    elif last_activation == "PReLU":
        layers.append(nn.PReLU())

    return nn.Sequential(*layers)


######## OLD CODE ########

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)