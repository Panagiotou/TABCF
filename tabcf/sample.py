import torch
import dice_ml

import argparse
import warnings
import time
import joblib

from tqdm import tqdm

from utils_train import preprocess, TabularDataset, get_name_form_args, get_optim_params_form_args

from tabcf.vae.model import BBMLPCLF
from tabcf.latent_utils import get_input_generate, recover_data, split_num_cat_target
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from raiutils.exceptions import UserConfigValidationException


# from GMM.src.model import get_model
# from GMM.src.FamilyTypes import get_mixture_family_from_str

warnings.filterwarnings('ignore')


class Decoder_Black_Box(torch.nn.Module):
    def __init__(self, black_box_clf, pre_decoder, pre_encoder, token_dim, input_encoded_continuous_feature_indexes, input_encoded_categorical_feature_indexes):
        super(Decoder_Black_Box, self).__init__()
        self.black_box_clf = black_box_clf
        self.pre_decoder = pre_decoder
        self.pre_encoder = pre_encoder
        self.token_dim = token_dim
        self.input_encoded_continuous_feature_indexes =  input_encoded_continuous_feature_indexes
        self.num_numerical = len(input_encoded_continuous_feature_indexes)
        self.input_encoded_categorical_feature_indexes = input_encoded_categorical_feature_indexes

    def bb_only(self, input_data):
        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor
        # Pass concatenated output through pre_decoder
        output = self.black_box_clf(input_data)

        if original_input_dim == 1:
            return output[0]

        return output
    
    def encode_z(self, input_data):

        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        num_input_data = input_data[:, :self.num_numerical]
        # cat_input_data = input_data[:, self.num_numerical:]
        cat_input_data = []

        for idxs in self.input_encoded_categorical_feature_indexes:
            curr_split = input_data[:, idxs]
            label_value = torch.argmax(curr_split, dim=None, keepdim=False)
            cat_input_data.append(label_value)

        cat_input_data = torch.stack(cat_input_data)

        # Expand the dimensions of the 1D tensor to match the 2D tensor
        cat_input_data = cat_input_data.unsqueeze(0)

        # encoded = torch.concat((num_input_data, cat_input_data), dim=1)


        encoded = self.pre_encoder(num_input_data, cat_input_data)
        encoded = encoded[:, 1:, :]


        B, num_tokens, token_dim = encoded.size()
        in_dim = num_tokens * token_dim
        
        encoded = encoded.view(B, in_dim).squeeze().detach()
        return encoded


    def decode_z(self, input_data):
        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        input_data = input_data.reshape(input_data.shape[0], -1, self.token_dim)

        
        x_hat_num, x_hat_cat = self.pre_decoder(input_data)

        # print(x_hat_num)
        # for xx in x_hat_cat:
        #     print(xx)

        # exit(1)
        concatenated = torch.cat((x_hat_num, *x_hat_cat), dim=-1)

        return concatenated
    
    def forward(self, input_data):

        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        input_data = input_data.reshape(input_data.shape[0], -1, self.token_dim)

        
        x_hat_num, x_hat_cat = self.pre_decoder(input_data)

        # print(x_hat_num)
        # for xx in x_hat_cat:
        #     print(xx)

        # exit(1)
        concatenated = torch.cat((x_hat_num, *x_hat_cat), dim=-1)

        # Pass concatenated output through pre_decoder
        output = self.black_box_clf(concatenated)

        if original_input_dim == 1:
            return output[0]

        return output
    

def test_decoder_black_box(pre_decoder, decoder_black_box_clf, loaded_black_box_clf, train_z, data_dir, info, cat_encoding='one-hot'):

    X_num_cat_ohe, _, categories_ohe, d_numerical_ohe, y_ohe = preprocess(data_dir, task_type = info['task_type'], cat_encoding='one-hot', num_encoding='minmax')

    X_train_num_cat_ohe, X_test_num_cat_ohe = X_num_cat_ohe

    y_train_num_cat_ohe, y_test_num_cat_ohe = y_ohe


    train_data_clf_ohe = TabularDataset(X_train_num_cat_ohe, y=y_train_num_cat_ohe, info=info)

    train_loader_clf_ohe = DataLoader(
        train_data_clf_ohe,
        batch_size = 1,
        shuffle = False,
        num_workers = 4,
    )

    with torch.no_grad():
        for batch_num, batch_cat, batch_y in train_loader_clf_ohe:
            if cat_encoding=='one-hot':
                batch = torch.tensor(batch_num)
            else:
                batch = torch.cat((batch_num, batch_cat), dim=1)

            print("Input", batch)
            outputs = loaded_black_box_clf(batch)

            print("Prediction based on input", outputs)

            break

        latent_in = train_z[0]
        original_input_dim = latent_in.dim()

        if original_input_dim == 1:
            latent_in = latent_in.unsqueeze(0)  # Unsqueezes the tensor

        latent_in = latent_in.reshape(latent_in.shape[0], -1, info["token_dim"])
        x_hat_num, x_hat_cat = pre_decoder(latent_in)
        reconstructed = torch.cat((x_hat_num, *x_hat_cat), dim=-1)
        print("Reconstructed", reconstructed)


        differences = torch.abs(batch - reconstructed)
        difference_indices = (batch != reconstructed).nonzero(as_tuple=True)
        for idx in zip(*difference_indices):
            print(f"Index: {idx}, Difference: {differences[idx]}")

        outputs_rec = loaded_black_box_clf(reconstructed)
        print("Prediction based on reconstruction", outputs_rec)
        exit(1)




def main(args):


    dataname = args.dataname
    data_dir = f'data/{dataname}'

    device = args.device
    num_samples = args.num_samples
    save_path = args.save_path
    verbose = args.verbose

    # steps = args.steps
    # validity = args.validity
    # proximity = args.proximity
    # sparsity = args.sparsity
    # latent_clf = args.latent_clf
    # diffusion = args.diffusion
    immutable = args.immutable


    proximity_weight_input = args.proximity_weight_input
    proximity_weight_latent = args.proximity_weight_latent
    proximity_latent_loss = args.proximity_latent_loss
    plausibility_weight_latent = args.plausibility_weight_latent
    plausibility_latent_loss = args.plausibility_latent_loss

    hidden_dims = args.hidden_dims
    plot_gradients = args.plot_gradients


    optimization_parameters = get_optim_params_form_args(args) 

    dice_optimization = True
    dice_method = args.dice_method

    dice_post_hoc_sparsity = args.dice_post_hoc_sparsity
    decode_before_loss = args.decode_before_loss
    num_encoding = args.num_encoding


    total_CFs = args.total_CFs
    min_iter = args.min_iter
    max_iter = args.max_iter

    proximity_input_loss = args.proximity_input_loss
    
    output_file_name_vae = get_name_form_args(args)


    output_folder_name = f'{dataname}/TABCF/{output_file_name_vae}/{dice_method}/{optimization_parameters}/'

    if num_samples > 0:
        output_folder_name = f'{output_folder_name}/{num_samples}_samples'
    else:
        output_folder_name = f'{output_folder_name}/all_samples'

    if total_CFs > 1:
        output_folder_name = f'{output_folder_name}_CFs_{total_CFs}'

    save_path = os.path.join(save_path, output_folder_name)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if os.listdir(save_path):
            print("CFs for setting", output_folder_name, "already computed")
            exit(0)



    train_z, test_z, _, _, ckpt_path, vae_ckpt_dir, info, num_inverse, cat_inverse, y = get_input_generate(args)

    X_num, _, _, _, _ = preprocess(data_dir, task_type = info['task_type'], num_encoding=num_encoding)

    num_numerical = X_num[0].shape[-1]

    train_set_path = f'data/{dataname}/train.csv'
    test_set_path = f'data/{dataname}/test.csv'

    train_df = pd.read_csv(train_set_path)
    test_df = pd.read_csv(test_set_path)


    target_column = info["target_col"]
    negative_class = info["negative_class"]
    target_class = info["target_class"]
    num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
    cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]
    
    column_names = info["column_names"]
    target_col = info["target_col"]
    column_names.remove(target_col)


    mean = train_z.mean(0)

    in_dim = train_z.shape[1] 
    in_dim_test = test_z.shape[1] 
    
    # model = Model(denoise_fn = denoise_fn_test, hid_dim = test_z.shape[1]).to(device)



    num_latent_features = train_z.shape[1]
    latent_column_names = [f'latent_{i}' for i in range(num_latent_features)]

    train_z_df = pd.DataFrame(train_z, columns=latent_column_names)

    test_z_df = pd.DataFrame(test_z, columns=latent_column_names)

    train_z_df[target_column] = train_df[target_column].map({target_class: 1, negative_class: 0})

    train_z_df_target = train_z_df[train_z_df[target_column]==1].reset_index(drop=True).drop(target_column, axis=1)

    X_num_cat_ohe, _, _, _, _ = preprocess(data_dir, task_type = info['task_type'], cat_encoding='one-hot', num_encoding='minmax')




    '''
        Generating samples    
    '''
    start_time = time.time()


    model_filename = f'{data_dir}/black_box_mlp.pkl'

    if hidden_dims is not None:
        model_filename = f'{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl'

        
    input_shape = len(info["num_col_idx"]) + len(info["cat_col_idx"])
        
    if dice_optimization:
        input_shape = X_num_cat_ohe[0].shape[-1]

    loaded_black_box_clf = BBMLPCLF(input_shape).to(device)

    loaded_black_box_clf.load_state_dict(torch.load(model_filename))
    # Ensure black_box_clf parameters are not updated
    for param in loaded_black_box_clf.parameters():
        param.requires_grad = False

    loaded_black_box_clf.eval()

    pre_decoder = info['pre_decoder'].to(device)
    pre_encoder = info['pre_encoder'].to(device)
    pre_decoder.eval()
    pre_encoder.eval()

    token_dim = info['token_dim']


    # test_decoder_black_box(pre_decoder, decoder_black_box_clf, loaded_black_box_clf, train_z, data_dir, info)

    ### Get only test_z that do not belong to the target class

    # if latent_clf:
    # with torch.no_grad():
        # test_outputs = loaded_black_box_clf(torch.tensor(test_z).to(device))
    # else:

    #---------------------- Input DiCE -------------------------------------


    train_df[target_column] = train_df[target_column].map({target_class: 1, negative_class: 0})

    data_dice_input = dice_ml.data.Data(dataframe=train_df, outcome_name=target_column, continuous_features=num_cols)

    dice_model_on_input = dice_ml.Model(loaded_black_box_clf.to("cpu"), backend='PYT', model_type='classifier', func="ohe-min-max", device=device)
    exp_input = dice_ml.Dice(data_dice_input, dice_model_on_input, method=dice_method)

    test_df_input = test_df.drop(target_column, axis=1)

    test_preds = np.argmax(dice_model_on_input.get_output(test_df_input), axis=1)


    feature_weights = exp_input.get_feature_weights(feature_weights="inverse_mad")

    input_minx, input_maxx, input_encoded_categorical_feature_indexes, input_encoded_continuous_feature_indexes, \
            input_cont_minx, input_cont_maxx, input_cont_precisions = data_dice_input.get_data_params_for_gradient_dice()

    #----------------------------------------------------------


    decoder_black_box_clf = Decoder_Black_Box(loaded_black_box_clf, pre_decoder, pre_encoder, token_dim, input_encoded_continuous_feature_indexes, input_encoded_categorical_feature_indexes).to(device)

    decoder_black_box_clf.eval()

    target_class_index = 1


    print("Loaded black box model from", model_filename)

    # DIFFERENT RUN

    indices_negative_class = np.where(test_preds != target_class_index)[0][:num_samples]


    negative_test_df = test_df.iloc[indices_negative_class].reset_index(drop=True)


    negative_test_z_df = test_z_df.iloc[indices_negative_class].reset_index(drop=True)


    negative_test_encoded = X_num_cat_ohe[1][indices_negative_class]

    if immutable:

        # Find the indices of the immutable features
        immutable_indices = [column_names.index(feature) for feature in info["immutable"]]

        immutable_mask = [i for i in range(token_dim * len(column_names))] 

        # Convert immutable_indices to a flat list of indices for the tensor
        immutable_flat_indices = []
        for idx in immutable_indices:
            start_index = idx * token_dim
            end_index = start_index + token_dim
            immutable_flat_indices.extend(range(start_index, end_index))

        # Ensure immutable_flat_indices are not in immutable_mask
        immutable_mask = list(set(immutable_mask) - set(immutable_flat_indices))

    else:
        immutable_mask = []


    dice_args = {"desired_class":1, "verbose":False, "total_CFs":total_CFs, "min_iter":min_iter, "max_iter":max_iter, "latent":True,"decode_before_loss": decode_before_loss, "proximity_weight":proximity_weight_input,
                 "proximity_weight_latent": proximity_weight_latent, "proximity_latent_loss": proximity_latent_loss, "plausibility_weight_latent":plausibility_weight_latent, "plausibility_latent_loss":plausibility_latent_loss,
                 "proximity_input_loss": proximity_input_loss, "immutable_mask": immutable_mask, "plot_gradients":plot_gradients}
    
    if decode_before_loss:
        dice_args["feature_weights"] = feature_weights
    if not dice_post_hoc_sparsity:
        dice_args["posthoc_sparsity_param"] = None


    print(dice_args)

    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()

    # data_latents = train_z.cpu().numpy()
    # print(data_latents.max(axis=1))

    all_cfs = []


    if dice_optimization:
        data_dice = dice_ml.data.Data(dataframe=train_z_df, outcome_name=target_column, continuous_features=latent_column_names)
        dice_model = dice_ml.Model(decoder_black_box_clf, backend='PYT', model_type='classifier', device=device)
        exp = dice_ml.Dice(data_dice, dice_model, method=dice_method, data_dice_input=data_dice_input, dice_model_on_input=dice_model_on_input, verbose=verbose, train_z_df_target=train_z_df_target)

        # for indx, row in tqdm(negative_test_z_df.iterrows(), total=len(negative_test_z_df), desc="CFs for test set (dice optimization)"):

        for indx, row in tqdm(negative_test_z_df.iterrows(), total=len(negative_test_z_df), desc="CFs for test set"):
            original_instance_input = negative_test_df.iloc[indx:indx+1]
            original_instance_z = negative_test_z_df.iloc[indx:indx+1]
            original_input_instance = negative_test_encoded[indx:indx+1]
            dice_args["original_input_instance"] = original_input_instance
            dice_args["index"] = indx
            
            try:
                dice_exp = exp.generate_counterfactuals(original_instance_z, **dice_args)
            except UserConfigValidationException as e: # cf not found, just add the original sample, and set target_prob to 0.0 
                # all_cfs.append(original_instance_input)
                print(e)
                # final_probs.append(0.0)
                continue


                
            cfs = dice_exp.cf_examples_list[0].final_cfs_df[column_names]#[latent_column_names]

            target_probs_final = dice_model_on_input.get_output(cfs)[:, target_class_index]
            cfs["id"] = indx
            cfs["target_prob"] = target_probs_final

            all_cfs.append(cfs)
    # profiler.stop()

    # profiler.print()

    if not decode_before_loss:

        syn_data = np.concatenate(all_cfs)

        syn_num, syn_cat = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

        syn_df = recover_data(syn_num, syn_cat, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)
    else:
        syn_df = pd.concat(all_cfs, ignore_index=True)




    syn_df.reset_index(drop=True, inplace=True)

    cols = ["id"] + info["column_names"]

    syn_df = syn_df[cols + ["target_prob"]]

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    negative_test_df["id"] = list(range(len(negative_test_df)))

    negative_test_df = negative_test_df[cols]
    
    negative_test_df.to_csv(os.path.join(save_path, "original_test_samples.csv"), index = False)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'