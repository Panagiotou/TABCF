import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time
import pandas as pd

from tabcf.vae.model import Model_VAE, Encoder_model, Decoder_model, BBMLPCLF, Encoder_model_Z

from utils_train import preprocess, TabularDataset, get_name_form_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch.optim as optim

LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2
    

def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z, gumbel_softmax=False, num_reconstr_loss="L2", device="cpu"):

    if X_num.shape[-1] == 0:
        num_loss = torch.tensor([0]).to(device)
    else:
        if num_reconstr_loss=="L2":
            num_loss = (X_num - Recon_X_num).pow(2).mean()
        else:
            num_loss = (X_num - Recon_X_num).abs().mean()
        
    cat_loss = 0
    acc = 0
    total_num = 0

    if gumbel_softmax:

        cat_loss_fn = nn.L1Loss()

        # cat_loss_fn = nn.CrossEntropyLoss()
        # cat_loss_fn = nn.NLLLoss()

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                # log_x_cat = torch.log(x_cat + 1e-9)
                # cat_loss += cat_loss_fn(log_x_cat, X_cat[:, idx])
                # x_hat = log_x_cat
                # print(x_cat[0])
                # print(X_cat[0, idx])
                # exit(1)
                x_cat_ohe = torch.nn.functional.one_hot(X_cat[:, idx], num_classes=x_cat.shape[-1]).float()
                cat_loss += cat_loss_fn(x_cat, x_cat_ohe)
                # cat_loss += cat_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat

                x_hat_for_acc = x_hat.argmax(dim = -1)
            acc += (x_hat_for_acc == X_cat[:,idx]).float().sum()
            total_num += x_hat_for_acc.shape[0]
        cat_loss /= (idx + 1)
        acc /= total_num
    else:
        cat_loss_fn = nn.CrossEntropyLoss()

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                cat_loss += cat_loss_fn(x_cat, X_cat[:, idx])
                
                x_hat = x_cat.argmax(dim = -1)
            acc += (x_hat == X_cat[:,idx]).float().sum()
            total_num += x_hat.shape[0]
        cat_loss /= (idx + 1)
        acc /= total_num
    # loss = num_loss + cat_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return num_loss, cat_loss, loss_kld, acc

def concatenate_input_dims(x, tensor=False):
    shape = x.shape
    new_shape = shape[:-2] + (shape[-2] * shape[-1],)
    reshaped_x = x.reshape(new_shape)

    if tensor:
        return torch.tensor(reshaped_x)
    return reshaped_x


def train_black_box_clf_original_data(info, train_loader, test_loader=None, num_epochs=100, data_dir="", cat_encoding=None, input_shape=None, gumbel_softmax=True, hidden_dims=None):

    
    model_filename = f'{data_dir}/black_box_mlp.pkl'
    if cat_encoding=='one-hot-drop-bin':
        model_filename = f'{data_dir}/black_box_mlp_ohe_drop_bin.pkl'

    if not gumbel_softmax:
        model_filename = f'{data_dir}/black_box_mlp_no_gumbel_softmax.pkl'

    if hidden_dims is not None:
        model_filename = f'{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl'

    if os.path.isfile(model_filename): 
        print("\tBlack box CLF on original data is already trained")
        # for epoch in range(num_epochs):
        #     pbar = tqdm(train_loader, total=len(train_loader))
        #     pbar.set_description(f"Black Box Epoch {epoch+1}/{num_epochs}")

        #     for batch_num, batch_cat, batch_y in pbar:

        #         batch = torch.cat((batch_num, batch_cat), dim=1)

        #         print(batch[0])
        #         exit(1)
        return 
    
    input_shape_original = len(info["num_col_idx"]) + len(info["cat_col_idx"])

    if cat_encoding=='one-hot-drop-bin':
        print("Training DIFFERENTIABLE black box CLF on original data (one hot drop binary)...")
    else:
        print("Training DIFFERENTIABLE black box CLF on original data (one hot)...")
    print("Original input shape is", input_shape_original)

    if input_shape is not None:
        print("Given input shape is", input_shape)
    else:
        input_shape = input_shape_original 


    # class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}
    
    # y_n = np.array([class_mapping[class_name[0]] for class_name in y_train])


    # y_train_t = torch.tensor(y_n)
    

    clf = BBMLPCLF(input_shape, return_logits=True)

    criterion = nn.CrossEntropyLoss()  # Use Cross Entropy Loss for classification
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    for epoch in range(num_epochs):
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Black Box Epoch {epoch+1}/{num_epochs}")

            for batch_num, batch_cat, batch_y in pbar:

                if cat_encoding=='one-hot' or cat_encoding=='one-hot-drop-bin':
                    batch = torch.tensor(batch_num)
                else:
                    batch = torch.cat((batch_num, batch_cat), dim=1)

                clf.train()
                optimizer.zero_grad()
                outputs = clf(batch)
                loss = criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize


    if test_loader is not None:
        clf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in test_loader:
                if cat_encoding=='one-hot' or cat_encoding=='one-hot-drop-bin':
                    batch = torch.tensor(batch_num)
                else:
                    batch = torch.cat((batch_num, batch_cat), dim=1)
                outputs = clf(batch)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total

        print(f'\tInfo: Black box MLP CLF (input) accuracy on test set: {accuracy:.4f}')
    torch.save(clf.state_dict(), model_filename)


def train_black_box_clf_latent(X, y, info, X_test=None, y_test=None, ckpt_dir="" ):

    X = X[:, 1:, :]
    X = concatenate_input_dims(X, tensor=True)

    class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}
    
    y_n = np.array([class_mapping[class_name[0]] for class_name in y])


    y = torch.tensor(y_n)

    clf = BBMLPCLF(X.shape[-1])
    criterion = nn.CrossEntropyLoss()  # Use Cross Entropy Loss for classification
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = clf(X)  # Forward pass
        loss = criterion(outputs, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

    model_filename = f'{ckpt_dir}/black_box_latent.pt'

    torch.save(clf.state_dict(), model_filename)

    if X_test is not None:

        X_test = X_test[:, 1:, :]
        X_test = concatenate_input_dims(X_test, tensor=True)
    
        y_test_n = np.array([class_mapping[class_name[0]] for class_name in y_test])


        # Put the model in evaluation mode
        clf.eval()

        # Pass the test dataset through the clf to get predictions
        with torch.no_grad():
            test_outputs = clf(X_test)

        _, y_pred = torch.max(test_outputs, 1)

        acc = accuracy_score(y_test_n, y_pred)

        print(f'\tInfo: Black box MLP clf (latent) accuracy on test set: {acc:.4f}')


def main(args):


    dataname = args.dataname
    data_dir = f'data/{dataname}'

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    reparam = args.reparam

    #encoding
    num_encoding = args.num_encoding

    #loss
    num_reconstr_loss = args.num_reconstr_loss


    #activations
    sigmoid = args.sigmoid
    gumbel_softmax = args.gumbel_softmax
    tau = args.tau
    kl_weight = args.kl_weight

    hidden_dims = args.hidden_dims

    output_file_name_vae = get_name_form_args(args)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/{output_file_name_vae}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        print("VAE for setting", output_file_name_vae, "already exists")
        # exit(0)



    device =  args.device
    batch_size = 4096


    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    




    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical, y = preprocess(data_dir, task_type = info['task_type'], num_encoding=num_encoding)





    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat


    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    y_train, y_test = y

    train_data_clf = TabularDataset(X_train_num.float(), X_cat=X_train_cat, y=y_train, info=info)
    test_data_clf = TabularDataset(X_test_num.float(), X_cat=X_test_cat, y=y_test, info=info)

    #----------------------------------------------------------------------------

    X_num_cat_ohe, _, categories_ohe, d_numerical_ohe, y_ohe = preprocess(data_dir, task_type = info['task_type'], cat_encoding='one-hot', num_encoding='minmax')
    X_num_cat_ohe_drop_bin, _, _, _, y_ohe_drop_bin = preprocess(data_dir, task_type = info['task_type'], cat_encoding='one-hot-drop-bin', num_encoding='minmax')

    X_train_num_cat_ohe, X_test_num_cat_ohe = X_num_cat_ohe

    y_train_num_cat_ohe, y_test_num_cat_ohe = y_ohe

    X_train_num_cat_ohe_drop_bin, X_test_num_cat_ohe_drop_bin = X_num_cat_ohe_drop_bin

    y_train_num_cat_ohe_drop_bin, y_test_num_cat_ohe_drop_bin = y_ohe_drop_bin

    train_data_clf_ohe = TabularDataset(X_train_num_cat_ohe, y=y_train_num_cat_ohe, info=info)
    
    train_data_clf_ohe_drop_bin = TabularDataset(X_train_num_cat_ohe_drop_bin, y=y_train_num_cat_ohe_drop_bin, info=info)

    train_loader_clf_ohe = DataLoader(
        train_data_clf_ohe,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    train_loader_clf_ohe_drop_bin = DataLoader(
        train_data_clf_ohe_drop_bin,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )


    test_data_clf_ohe = TabularDataset(X_test_num_cat_ohe, y=y_test_num_cat_ohe, info=info)

    test_data_clf_ohe_drop_bin = TabularDataset(X_test_num_cat_ohe_drop_bin, y=y_test_num_cat_ohe_drop_bin, info=info)

    test_loader_clf_ohe = DataLoader(
        test_data_clf_ohe,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    test_loader_clf_ohe_drop_bin = DataLoader(
        test_data_clf_ohe_drop_bin,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    #----------------------------------------------------------------------------


    train_data = TabularDataset(X_train_num.float(), X_cat=X_train_cat)


    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    train_loader_clf = DataLoader(
        train_data_clf,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    test_loader_clf = DataLoader(
        test_data_clf,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )


    train_black_box_clf_original_data(info, train_loader_clf_ohe, test_loader=test_loader_clf_ohe, data_dir=data_dir, num_epochs=100, cat_encoding='one-hot', input_shape=X_train_num_cat_ohe.shape[-1], gumbel_softmax=gumbel_softmax, hidden_dims=hidden_dims)
    
    # train_black_box_clf_original_data(info, train_loader_clf_ohe_drop_bin, test_loader=test_loader_clf_ohe_drop_bin, data_dir=data_dir, num_epochs=100, cat_encoding='one-hot-drop-bin', input_shape=X_train_num_cat_ohe_drop_bin.shape[-1])


    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True, gumbel_softmax=gumbel_softmax, tau=tau, sigmoid=sigmoid)
    model = model.to(device)

    if reparam:
        pre_encoder = Encoder_model_Z(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    else:
        pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
        
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, gumbel_softmax=gumbel_softmax, sigmoid=sigmoid).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    num_epochs = 4000
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta

    if kl_weight > 0 :
        beta = 1/num_epochs

    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        # if kl_weight > 0:
        #     beta = epoch * kl_weight / num_epochs 

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z, gumbel_softmax=gumbel_softmax, num_reconstr_loss=num_reconstr_loss, device=device)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z, gumbel_softmax=gumbel_softmax)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if kl_weight < 0:
                            if beta > min_beta:
                                beta = beta * lambd
                    else:
                        if beta < kl_weight:
                            beta = beta * 1.3



        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))


    # Define the file path where you want to save the log
    log_file_path = f'{ckpt_dir}/final_log.txt'

    # Open the file in append mode to add new logs at the end of the file
    with open(log_file_path, 'a') as log_file:
        log_file.write('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}\n'.format(
            epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item()))
    
    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()


        test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)
        
        np.save(f'{ckpt_dir}/test_z.npy', test_z)

        print('Successfully save pretrained embeddings in disk!')


    # train_black_box_clf_latent(train_z, y_train, info, X_test=test_z, y_test=y_test, ckpt_dir=ckpt_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'