#Filename:	VariationalAutoencoder.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 01 Jan 2021 10:31:50  WIB
import os
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

from carla import log
from carla.recourse_methods.autoencoder.save_load import get_home

tf.compat.v1.disable_eager_execution()
import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):

    def __init__(self, data_name: str, layers: List, mutable_mask):

        super(VariationalAutoencoder, self).__init__()

        self._data_name = data_name

        self.data_size = layers[0]
        self.encoded_size = layers[-1]
        self.hidden_dims = layers[1:-2]

        modules = []
        
        in_channels = self.data_size
        #create encoder module
        for h_dim in self.hidden_dims:
            modules.append(
                    nn.Sequential(
                        nn.Linear(in_channels, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.Dropout(0.1),
                        nn.ReLU(),
                        )
                    )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(12, self.encoded_size)
        self.fc_var = nn.Linear(12, self.encoded_size)

        #create decoder module
        modules = []
        in_channels = self.encoded_size

        for h_dim in reversed(self.hidden_dims):
            modules.append(
                    nn.Sequential(
                        nn.Linear(in_channels, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.Dropout(0.1),
                        nn.ReLU(),
                        )
                    )
            in_channels = h_dim

        modules.append(nn.Linear(in_channels, self.data_size))
        self.sig = nn.Sigmoid()
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, input_x):

        output = self.encoder(input_x)
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)

        return [mu, log_var]

    def decode(self, z):

        x = self.decoder(z)
        for v in self.data_interface.encoded_categorical_feature_indices:    
            start_index = v[0]
            end_index = v[-1] + 1
            x[:,start_index:end_index] = self.sig(x[:,start_index:end_index])
        return x
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_x):

        mu, log_var =  self.encode(input_x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input_x, mu, log_var]

    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=32,
        ):
        
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        self.epochs = epochs
        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )
        # criterion = nn.BCELoss(reduction="sum")
        criterion = nn.BCELoss()
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        log.info("Start training of Variational Autoencoder...")
        for epoch in range(epochs):

            beta = epoch * kl_weight / epochs

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0

            # Train for all the batches
            for data in train_loader:
                data = data.view(data.shape[0], -1)
                data = data.to(self.device).float()

                # forward pass
                reconstruction, mu, log_var = self(data)



                recon_loss = criterion(reconstruction, data)

                kld_loss = self.kld(mu, log_var)

                loss = recon_loss + beta * kld_loss
                # print(data.shape)
                # print(reconstruction)
                # print(data)
                # print(recon_loss)
                # print(kld_loss)
                # print(loss)
                # exit(1)

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            ELBO[epoch] = train_loss / train_loss_num
            if epoch % 10 == 0:
                log.info(
                    "[Epoch: {}/{}] [objective: {:.3f}]".format(
                        epoch, epochs, ELBO[epoch, 0]
                    )
                )

            ELBO_train = ELBO[epoch, 0].round(2)
            log.info("[ELBO train: " + str(ELBO_train) + "]")

        self.save()
        log.info("... finished training of Variational Autoencoder.")

        self.eval()

    def compute_loss(self, output, input_x, mu, log_var):

        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        con_criterion = nn.MSELoss()
        cat_criterion = nn.BCELoss()
        
        cat_loss = 0
        con_loss = 0
        
        for v in self.data_interface.encoded_categorical_feature_indices:
            start_index = v[0]
            end_index = v[-1]+1
            cat_loss += cat_criterion(output[:, start_index:end_index], input_x[:, start_index:end_index])
        
        categorial_indices = []
        for v in self.data_interface.encoded_categorical_feature_indices:
            categorial_indices.extend(v)

        continuous_indices = list(set(range(36)).difference(categorial_indices))
        con_loss = con_criterion(output[:, continuous_indices], input_x[:, continuous_indices])
        recon_loss = torch.mean(cat_loss + con_loss) 
        total_loss = kl_loss + recon_loss

        return total_loss, recon_loss, kl_loss

    def trained_exists(self, input_shape, epochs):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "{}_{}_epochs_{}.{}".format(self._data_name, input_shape, epochs, "pt"),
        )
        return os.path.isfile(load_path)


    def load(self, input_shape, epochs):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "{}_{}_epochs_{}.{}".format(self._data_name, input_shape, epochs, "pt"),
        )

        self.load_state_dict(torch.load(load_path))

        self.eval()
        print("Loaded VAE from", load_path)

        return self

    def save(self):
        cache_path = get_home()

        save_path = os.path.join(
            cache_path,
            "{}_{}_epochs_{}.{}".format(self._data_name, self._input_dim, self.epochs, "pt"),
        )

        torch.save(self.state_dict(), save_path)