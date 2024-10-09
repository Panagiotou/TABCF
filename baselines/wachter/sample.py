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
from carla.data.catalog import CsvCatalog, OnlineCatalog, TrainTestCsvCatalog
from carla import MLModelCatalog

from carla import MLModel
from carla.recourse_methods import Wachter
import yaml

# Custom black-box models need to inherit from
# the MLModel interface
class MyOwnModel(MLModel):
    def __init__(self, data, data_dir, info, input_shape=66, device="cpu", encoding_method="OneHot", hidden_dims=None):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model

        if encoding_method=="OneHot":
            loaded_black_box_clf = BBMLPCLF(input_shape).to(device)
            model_filename = f'{data_dir}/black_box_mlp.pkl'
            if hidden_dims is not None:
                model_filename = f'{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl'
        else:
            input_shape = 64
            loaded_black_box_clf = BBMLPCLF(input_shape).to(device)
            model_filename = f'{data_dir}/black_box_mlp_ohe_drop_bin.pkl'

        self.input_shape = input_shape

        loaded_black_box_clf.load_state_dict(torch.load(model_filename))

        self._mymodel = loaded_black_box_clf
        self._mymodel.eval()
        self.info = info
        self.device = device

        input_cols = list(data.df.columns)
        input_cols.remove(info["target_col"])
        self.input_cols = input_cols

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.input_cols

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).float().to(self.device)
            return self._mymodel(x).detach().cpu().numpy()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float().to(self.device)
            return self._mymodel(x).detach().cpu().numpy()
        
        return self._mymodel(x)

def main(args):
    # ML_modelpath = helpers.get_adult_income_modelpath(backend="PYT")
    # # Step 2: dice_ml.Model
    # # m = dice_ml.Model(model_path=ML_modelpath, backend="PYT", func="ohe-min-max")
    # model = torch.load(ML_modelpath)

    OneHot_drop_binary = False

    if OneHot_drop_binary:
        encoding_method = "OneHot_drop_binary"
    else:
        encoding_method = "OneHot"


    dataname = args.dataname
    method = args.method
    data_dir = f'data/{dataname}'

    device = args.device
    num_samples = args.num_samples
    verbose = args.verbose
    save_path = args.save_path
    hidden_dims = args.hidden_dims

    _, _, _, _, _, _, info, num_inverse, cat_inverse, y = get_input_generate(args)

    num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
    cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]
    immutable = []

    output_folder_name = f'{dataname}/Wachter/'

    if num_samples > 0:
        output_folder_name = f'{output_folder_name}/{num_samples}_samples'
    else:
        output_folder_name = f'{output_folder_name}/all_samples'

    save_path = os.path.join(save_path, output_folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if os.listdir(save_path):
            print("CFs for setting", output_folder_name, "already computed")
            exit(0)


    train_set_path = f'data/{dataname}/train.csv'
    test_set_path = f'data/{dataname}/test.csv'


    dataset = TrainTestCsvCatalog(train_path=train_set_path,
                        test_path=test_set_path,
                        continuous=num_cols,
                        categorical=cat_cols,
                        immutables=immutable,
                        target=info["target_col"],
                        encoding_method=encoding_method)
    
    

    test_df_encoded = dataset.df_test

    data_name = dataname

    X_num_cat_ohe, _, _, _, _ = preprocess(data_dir, task_type = info['task_type'], cat_encoding='one-hot', num_encoding='minmax')

    input_shape = X_num_cat_ohe[0].shape[-1]

    model = MyOwnModel(dataset, data_dir, info, input_shape=input_shape, device=args.device, encoding_method=encoding_method, hidden_dims=hidden_dims)

    with open("baselines/CARLA/experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    hyperparameters = setup_catalog["recourse_methods"]["wachter"]["hyperparams"]
    hyperparameters["data_name"] = data_name


    # hyperparameters["vae_params"]["layers"] = [
    #     sum(model.get_mutable_mask())
    # ] + hyperparameters["vae_params"]["layers"]


    hyperparameters["encoding_method"] = encoding_method

    hyperparameters["epochs"] = 500
    # hyperparameters["vae_params"]["batch_size"] = 1000

    
    hyperparameters["negative_class"] = info["negative_class"]
    hyperparameters["target_class"] = info["target_class"]

    print(hyperparameters)

    categories = dataset.encoder.categories_

    continuous_feature_indexes = list(range(0, len(dataset.continuous)))

    categorical_feature_indexes = []
    current_index = len(dataset.continuous)

    for category in categories:
        if len(category)==2 and encoding_method=="OneHot_drop_binary":
            categorical_feature_indexes.append([current_index])
            current_index += 1
        else:
            categorical_feature_indexes.append(list(range(current_index, current_index + len(category))))
            current_index += len(category)


    # print(continuous_feature_indexes)
    # print(categorical_feature_indexes)
    # print(categories)
    # exit(1)

    hyperparameters["continuous_feature_indexes"] = continuous_feature_indexes
    hyperparameters["categorical_feature_indexes"] = categorical_feature_indexes

    hyperparameters["binary_cat_features"] = True
    hyperparameters["device"] = device



    column_names = info["column_names"]
    target_col = info["target_col"]
    column_names.remove(target_col)
    target_class_index = 1

    test_preds = np.argmax(model.predict_proba(test_df_encoded.drop(target_col, axis=1)), axis=1)
    indices_negative_class = np.where(test_preds != target_class_index)[0][:num_samples]
    factuals = test_df_encoded.iloc[indices_negative_class]

    # data_name = "adult"
    # dataset = OnlineCatalog(data_name, encoding_method="OneHot")

    # ml_model = MLModelCatalog(
    #     dataset,
    #     model_type="ann",
    #     # load_online=True,
    #     backend="pytorch",
    # )
    # hyperparameters = {
    #     "data_name": data_name,
    #     "vae_params": {
    #         "layers": [sum(ml_model.get_mutable_mask()), 512, 256, 8],
    #     },
    # }

    # print(dataset.df.shape)

    # hyperparameters = {
    #     "data_name": data_name,
    #     "n_search_samples": 100,
    #     "p_norm": 1,
    #     "step": 0.1,
    #     "max_iter": 1000,
    #     "clamp": True,
    #     "binary_cat_features": True,
    #     "vae_params": {
    #         "layers": [sum(ml_model.get_mutable_mask()), 64, 8],
    #         "train": True,
    #         "lambda_reg": 1e-6,
    #         "epochs": 50,
    #         "lr": 1e-3,
    #         "batch_size": 32,
    #     },
    # }
    # rv = CCHVAE(ml_model, hyperparameters)

    # exit(1)


    rv = Wachter(model, hyperparameters)


    counterfactuals_enc = rv.get_counterfactuals(factuals)
    
    counterfactuals_enc = counterfactuals_enc.reset_index(drop=True)

    # nan_indices = counterfactuals_enc[counterfactuals_enc.isna().any(axis=1)].index

    # not_nan_indices = counterfactuals_enc.dropna().index

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    counterfactuals_enc_found = counterfactuals_enc.dropna()

    # print(counterfactuals_enc_found)

    target_probs_final = model.predict_proba(counterfactuals_enc_found)[:, target_class_index]


    counterfactuals_found = dataset.inverse_transform(counterfactuals_enc_found)



    counterfactuals_found["target_prob"] = target_probs_final
    counterfactuals_found["id"] = list(counterfactuals_enc_found.index)


    cols = ["id"] + column_names

    syn_df = counterfactuals_found[cols + ["target_prob"]]

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index = False)
    
    test_df = pd.read_csv(test_set_path)

    negative_test_df = test_df.iloc[indices_negative_class].reset_index(drop=True)

    negative_test_df["id"] = list(range(len(negative_test_df)))

    negative_test_df = negative_test_df[cols]
    
    negative_test_df.to_csv(os.path.join(save_path, "original_test_samples.csv"), index = False)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation DiCE')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'