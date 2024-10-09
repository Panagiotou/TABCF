import os
import json
import numpy as np
import pandas as pd
import torch
from utils_train import preprocess, get_name_form_args
from tabcf.vae.model import Decoder_model, Encoder_model, Encoder_model_Z
import torch.nn.functional as F

def get_input_train(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


def get_input_generate(args):
    dataname = args.dataname
    gumbel_softmax = args.gumbel_softmax
    num_encoding = args.num_encoding

    #activations
    sigmoid = args.sigmoid
    gumbel_softmax = args.gumbel_softmax
    tau = args.tau

    output_file_name_vae = get_name_form_args(args)



    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    decoder_save_path = f'{curr_dir}/vae/ckpt/{dataname}/{output_file_name_vae}/decoder.pt'
    encoder_save_path = f'{curr_dir}/vae/ckpt/{dataname}/{output_file_name_vae}/encoder.pt'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']


    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'

    vae_ckpt_dir = f'{curr_dir}/vae/ckpt/{dataname}/{output_file_name_vae}'

    _, _, categories, d_numerical, num_inverse, cat_inverse, y = preprocess(dataset_dir, task_type = task_type, inverse = True, num_encoding=num_encoding)

    if args.method == "tabcf":
        embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/{output_file_name_vae}/train_z.npy'
        embedding_save_path_test = f'{curr_dir}/vae/ckpt/{dataname}/{output_file_name_vae}/test_z.npy'
        train_z = torch.tensor(np.load(embedding_save_path)).float()
        test_z = torch.tensor(np.load(embedding_save_path_test)).float()


        train_z = train_z[:, 1:, :]
        test_z = test_z[:, 1:, :]


        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim
    
        train_z = train_z.view(B, in_dim)

        B_test, num_tokens_test, token_dim_test = test_z.size()
        in_dim_test = num_tokens_test * token_dim_test
    
        test_z = test_z.view(B_test, in_dim_test)


        pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32, gumbel_softmax=gumbel_softmax, sigmoid=sigmoid, tau=tau)


        if args.reparam:
            pre_encoder = Encoder_model_Z(2, d_numerical, categories, 4, n_head = 1, factor = 32)
        else:
            pre_encoder = Encoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32)


        pre_decoder.load_state_dict(torch.load(decoder_save_path))

        info['pre_decoder'] = pre_decoder

        pre_encoder.load_state_dict(torch.load(encoder_save_path))

        info['pre_encoder'] = pre_encoder

        info['token_dim'] = token_dim

        return train_z, test_z, curr_dir, dataset_dir, ckpt_dir, vae_ckpt_dir, info, num_inverse, cat_inverse, y
    
    else:
        return None, None, curr_dir, dataset_dir, ckpt_dir, vae_ckpt_dir, info, num_inverse, cat_inverse, y

def decode_input(syn_data, info, num_inverse, cat_inverse, inverse_num=False):
    device = syn_data.device
    token_dim = info['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

    pre_decoder = info['pre_decoder'].to(device)

    pre_decoder.eval()


    norm_input = pre_decoder(syn_data)


    x_hat_num, x_hat_cat = norm_input


    syn_cat = []
    for pred in x_hat_cat:
        max_indices = pred.argmax(dim=-1)
        ohe_pred = F.one_hot(max_indices, num_classes=pred.size(-1))
        softmax_pred = F.softmax(pred)

        gumbel_softmax_pred = F.gumbel_softmax(pred, tau=1, hard=True)
        syn_cat.extend(gumbel_softmax_pred)
        # syn_cat.append(ohe_pred)
    syn_num = x_hat_num
    syn_cat = torch.cat(syn_cat).unsqueeze(0)

    # syn_cat = torch.stack(syn_cat).t()
    # print(syn_cat.shape)

    if inverse_num:
        syn_num = num_inverse(syn_num)

    # print(syn_num.shape)
    # exit(1)

    print(syn_num)
    print(syn_cat)

    decoded = torch.cat((syn_num, syn_cat), dim=1)
    print(decoded)
    exit(1)

    return decoded
 
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    pre_decoder = info['pre_decoder'].to(device)
    token_dim = info['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

    norm_input = pre_decoder(torch.tensor(syn_data).to(device))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)


    return syn_num, syn_cat

def recover_data(syn_num, syn_cat, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()


    for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
        if i in set(num_col_idx):
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
    return syn_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat

