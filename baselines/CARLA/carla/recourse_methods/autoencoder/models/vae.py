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


class VariationalAutoencoder(nn.Module):
    def __init__(self, data_name: str, layers: List, mutable_mask, continuous_feature_indexes=None, categorical_feature_indexes=None, encoding_method="OneHot"):
        """

        Parameters
        ----------
        data_name:
            Name of the dataset, used for the name when saving and loading the model.
        layers:
            List of layer sizes.
        mutable_mask:
            Mask that indicates which feature columns are mutable, and which are immutable. Setting
            all columns to mutable, results in the standard case.
        """
        super(VariationalAutoencoder, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )
        self.continuous_feature_indexes = continuous_feature_indexes
        self.categorical_feature_indexes = categorical_feature_indexes
        self.encoding_method = encoding_method

        self._data_name = data_name
        self._input_dim = layers[0]
        latent_dim = layers[-1]

        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.ReLU())
        encoder = nn.Sequential(*lst_encoder)

        self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        # the decoder does use the immutables, so need to increase layer size accordingly.
        layers[-1] += np.sum(~mutable_mask)

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append((nn.ReLU()))
        decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(
            decoder,
            nn.Linear(layers[1], self._input_dim),
            nn.Sigmoid(),
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.mutable_mask = mutable_mask

    def encode(self, x):
        return self._mu_enc(x), self._log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z)

    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]
        # the mutable part gets encoded
        mu_z, log_var_z = self.encode(x_mutable)
        z = self.__reparametrization_trick(mu_z, log_var_z)
        # concatenate the immutable part to the latents and decode both
        z = torch.cat([z, x_immutable], dim=-1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon

        return x, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)

    def kld(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=32,
        x_test=None
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        if isinstance(x_test, pd.DataFrame):
            x_test = torch.tensor(x_test.values).float().to(self.device)


        print("Train data shape", xtrain.shape)

        self.epochs = epochs
        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )

        criterion = nn.BCELoss(reduction="sum")
        cat_criterion = nn.CrossEntropyLoss()
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

            train_acc = 0
            total_num = 0
            
            # Train for all the batches
            for data in train_loader:

                self.train()
                optimizer.zero_grad()

                data = data.view(data.shape[0], -1)
                data = data.to(self.device).float()

                # forward pass
                reconstruction, mu, log_var = self(data)

                reconstruction_num = reconstruction[:, self.continuous_feature_indexes]
                data_num = data[:, self.continuous_feature_indexes]

                
                
                num_mse = (reconstruction_num - data_num).pow(2).mean()
                # cat_loss = 0

                if self.categorical_feature_indexes is not None:
                    for idxs in self.categorical_feature_indexes:

                        # if len(idxs) == 1 and self.encoding_method=="OneHot_drop_binary":
                        #     data_hat = data[:, idxs[0]]
                        #     reconstruction_hat = reconstruction[:, idxs[0]]
                        # else:
                        data_hat = torch.argmax(data[:, idxs], axis=1)
                        reconstruction_hat = torch.argmax(reconstruction[:, idxs], axis=1)

                        # cat_loss += cat_criterion(reconstruction[:, idxs], data_hat)
                        train_acc += (data_hat == reconstruction_hat).float().sum()
                        total_num += data_hat.shape[0]


                recon_loss = criterion(reconstruction, data)
                # recon_loss = num_loss + cat_loss

                kld_loss = self.kld(mu, log_var)

                loss = recon_loss + beta * kld_loss


                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()



                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            train_acc /= total_num

            '''
                Evaluation
            '''

            if x_test is not None:
                self.eval()

                with torch.no_grad():
                    reconstruction_val, _, _ = self(x_test)

                    recon_loss_val = criterion(reconstruction_val, x_test)
                    
                    reconstruction_num_val = reconstruction_val[:, self.continuous_feature_indexes]
                    data_num_val = x_test[:, self.continuous_feature_indexes]

                    num_mse_val = (reconstruction_num_val - data_num_val).pow(2).mean()

                    val_acc = 0
                    total_num_val = 0
                    for idxs in self.categorical_feature_indexes:
                        data_hat_val = torch.argmax(x_test[:, idxs], axis=1)
                        reconstruction_hat_val = torch.argmax(reconstruction_val[:, idxs], axis=1)

                        # cat_loss += cat_criterion(reconstruction[:, idxs], data_hat)
                        val_acc += (data_hat_val == reconstruction_hat_val).float().sum()
                        total_num_val += data_hat_val.shape[0]
                    
                    val_acc /= total_num_val

            elbo_train = train_loss / train_loss_num
            
            if x_test is not None:
                print('epoch: {}, beta = {:.6f}, Train ELBO {:.6f}, Train MSE: {:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, elbo_train, num_mse, kld_loss, num_mse_val.item(), train_acc.item(), val_acc.item() ))
            else:
                print('epoch: {}, beta = {:.6f}, Train ELBO {:.6f}, Train MSE: {:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, elbo_train, num_mse, kld_loss, train_acc.item()))

            # ELBO[epoch] = train_loss / train_loss_num
            # if epoch % 10 == 0:
            #     log.info(
            #         "[Epoch: {}/{}] [objective: {:.3f}]".format(
            #             epoch, epochs, ELBO[epoch, 0]
            #         )
            #     )

            # ELBO_train = ELBO[epoch, 0].round(2)
            # log.info("[ELBO train: " + str(ELBO_train) + "]")

        self.save()
        log.info("... finished training of Variational Autoencoder.")

        self.eval()

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
