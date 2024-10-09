import dice_ml
from dice_ml.utils import helpers # helper functions
import torch
import argparse
import warnings
import time
import joblib
import torch.nn.functional as F

from tqdm import tqdm
from contextlib import redirect_stdout
import io
import sys

from utils_train import preprocess, TabularDataset, get_name_form_args

from tabcf.vae.model import BBMLPCLF
from tabcf.latent_utils import get_input_generate, recover_data, split_num_cat_target
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from dice_ml.utils import helpers

warnings.filterwarnings('ignore')
from raiutils.exceptions import UserConfigValidationException


def main(args):
    # ML_modelpath = helpers.get_adult_income_modelpath(backend="PYT")
    # # Step 2: dice_ml.Model
    # # m = dice_ml.Model(model_path=ML_modelpath, backend="PYT", func="ohe-min-max")
    # model = torch.load(ML_modelpath)

    # print(model)
    # exit(1)

    dataname = args.dataname
    data_dir = f'data/{dataname}'

    device = args.device
    num_samples = args.num_samples
    verbose = args.verbose
        
    save_path = args.save_path
    dice_method = args.dice_method
    dice_post_hoc_sparsity = args.dice_post_hoc_sparsity

    proximity_weight_input = args.proximity_weight_input

    total_CFs = args.total_CFs
    min_iter = args.min_iter
    max_iter = args.max_iter
    hidden_dims = args.hidden_dims

    plot_gradients = args.plot_gradients

    optimization_parameters = f"proximity_weight_input[{proximity_weight_input}]_hidden_dims[{hidden_dims}]"


    
    output_folder_name = f'{dataname}/DiCE/{dice_method}/{optimization_parameters}'

    if num_samples > 0:
        output_folder_name = f'{output_folder_name}/{num_samples}_samples'
    else:
        output_folder_name = f'{output_folder_name}/all_samples'
        
    if total_CFs > 1:
        output_folder_name = f'{output_folder_name}_CFs_{total_CFs}'

    # if validity:
    #     output_folder_name += "_validity"
    # if proximity:
    #     output_folder_name += "_proximity"
    # if sparsity:
    #     output_folder_name += "_sparsity"

    save_path = os.path.join(save_path, output_folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _, _, _, _, _, _, info, num_inverse, cat_inverse, y = get_input_generate(args)

    target_class = info["target_class"]
    negative_class = info["negative_class"]

    model_filename = f'{data_dir}/black_box_mlp.pkl'

    if hidden_dims is not None:
        model_filename = f'{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl'

    # input_shape = len(info["num_col_idx"]) + len(info["cat_col_idx"])

    X_num_cat_ohe, _, _, _, _ = preprocess(data_dir, task_type = info['task_type'], cat_encoding='one-hot', num_encoding='minmax')

    input_shape = X_num_cat_ohe[0].shape[-1]


    loaded_black_box_clf = BBMLPCLF(input_shape).to(device)

    loaded_black_box_clf.load_state_dict(torch.load(model_filename))
    # Ensure black_box_clf parameters are not updated
    for param in loaded_black_box_clf.parameters():
        param.requires_grad = False

    loaded_black_box_clf.eval()


    target_class_index = 1

    train_set_path = f'data/{dataname}/train.csv'
    test_set_path = f'data/{dataname}/test.csv'

    train_df = pd.read_csv(train_set_path)
    test_df = pd.read_csv(test_set_path)

    print("Loaded black box model from", model_filename)


    # input_col_order = info["input_col_order"]
    target_column = info["target_col"]
    column_names = info["column_names"]
    target_col = info["target_col"]
    print(target_col)
    print(column_names)
    column_names.remove(target_col)

    num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
    cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

  


    #-------------------------------- DiCE stuff -------------------------------------------

    dice_args = {"desired_class":1, "verbose":False, "total_CFs":total_CFs, "proximity_weight":proximity_weight_input, "min_iter":min_iter, "max_iter":max_iter, "plot_gradients":plot_gradients}

    if not dice_post_hoc_sparsity:
        dice_args["posthoc_sparsity_param"] = None

    print("DiCE Args", dice_args)

    train_df[target_column] = train_df[target_column].map({target_class: 1, negative_class: 0})


    data_dice = dice_ml.data.Data(dataframe=train_df, outcome_name=target_column, continuous_features=num_cols)
    dice_model = dice_ml.Model(loaded_black_box_clf, backend='PYT', model_type='classifier', func="ohe-min-max", device=device)
    exp = dice_ml.Dice(data_dice, dice_model, method=dice_method, verbose=verbose)

    #-----------------------------------------------------------------------------------------
    test_df = test_df.drop(target_column, axis=1)

    test_preds = np.argmax(dice_model.get_output(test_df), axis=1)

    indices_negative_class = np.where(test_preds != target_class_index)[0][:+num_samples]

    negative_test_df = test_df.iloc[indices_negative_class].reset_index(drop=True)

    '''
        Generating samples    
    '''
    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()


    start_time = time.time()



    all_cfs = []

    for indx, row in tqdm(negative_test_df.iterrows(), total=len(negative_test_df), desc="CFs for test set"):

        original_instance = negative_test_df.iloc[indx:indx+1]

        dice_args["index"] = indx

        try:
            dice_exp = exp.generate_counterfactuals(original_instance, **dice_args)
        except UserConfigValidationException as e: # cf not found, just add the original sample, and set target_prob to 0.0 
            # original_instance["id"] = indx
            # original_instance["target_prob"] = 0.0
            # all_cfs.append(original_instance)
            continue

        cfs = dice_exp.cf_examples_list[0].final_cfs_df[column_names]

        # cf = cf[input_col_order]
        target_probs_final = dice_model.get_output(cfs)[:, target_class_index]
        cfs["id"] = indx

        cfs["target_prob"] = target_probs_final

        all_cfs.append(cfs)

    # profiler.stop()

    # profiler.print()

    syn_df = pd.concat(all_cfs, ignore_index=True)

    cols = ["id"] + info["column_names"]

    syn_df = syn_df[cols + ["target_prob"]]

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    negative_test_df["id"] = list(range(len(negative_test_df)))

    negative_test_df = negative_test_df[cols]

    negative_test_df.to_csv(os.path.join(save_path, "original_test_samples.csv"), index = False)

    # print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation DiCE')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'