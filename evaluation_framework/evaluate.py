import joblib
import argparse
import os 
import pandas as pd
import numpy as np
import json
from utils_train import get_name_form_args, get_optim_params_form_args
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler


from sdmetrics.single_column import StatisticSimilarity
from sdmetrics.single_column import TVComplement
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

from sklearn.ensemble import IsolationForest

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
from joblib import dump, load
from pyod.models.dif import DIF 



# print(clf.score_samples([[0.1], [0], [90]]))



def get_outlier_detection_model(train_df, data_dir, info, target_only=False, negative_only=False):

    if target_only:
        model_filename = f'{data_dir}/outlier_model_eval_target_only.pkl'
    elif negative_only:
        model_filename = f'{data_dir}/outlier_model_eval_negative_only.pkl'
    else:
        model_filename = f'{data_dir}/outlier_model_eval.pkl'

    if os.path.isfile(model_filename): 
        with open(model_filename, 'rb') as file:
            clf_IF = pickle.load(file)
    else:
        num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
        cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

        numeric_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])


        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)])
        
        clf_IF = Pipeline(steps=[('preprocessor', transformations),
                    ('classifier', IsolationForest(random_state=42))])

        clf_IF.fit(train_df)
                
        with open(model_filename, 'wb') as file:
            pickle.dump(clf_IF, file)

    return clf_IF


def get_outlier_detection_model_pyod(train_df, data_dir, info, target_only=False, negative_only=False):

    if target_only:
        model_filename = f'{data_dir}/outlier_model_eval_target_only_pyod.joblib'
    elif negative_only:
        model_filename = f'{data_dir}/outlier_model_eval_negative_only_pyod.pkl'
    else:
        model_filename = f'{data_dir}/outlier_model_eval_pyod.joblib'

    if os.path.isfile(model_filename): 
        clf_IF = load(model_filename)
    else:
        num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
        cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

        numeric_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])


        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)], sparse_threshold=0)
        
        clf_IF = Pipeline(steps=[('preprocessor', transformations),
                    ('classifier', DIF(random_state=42))])

        clf_IF.fit(train_df)

        dump(clf_IF, model_filename)

    return clf_IF

def get_plausibility_stats(correct_cfs, train_set_path, info, data_dir, target_only=False, pyod=False, negative_only=False):


    metadata = info['metadata'].copy()
    metadata['columns'] = {info["column_names"][int(key)]: value for key, value in metadata['columns'].items()}


    train_df = pd.read_csv(train_set_path)


    if target_only:
        train_df = train_df[train_df[info["target_col"]]==info["target_class"]]

    if negative_only:
        train_df = train_df[train_df[info["target_col"]]==info["negative_class"]]

    train_df = train_df.drop(info["target_col"], axis=1)


    if pyod:
        clf = get_outlier_detection_model_pyod(train_df, data_dir, info, target_only=target_only, negative_only=negative_only)
    else:
        clf = get_outlier_detection_model(train_df, data_dir, info, target_only=target_only, negative_only=negative_only)


    metadata['columns'].pop(info["target_col"])

    qual_report = QualityReport()
    qual_report.generate(train_df, correct_cfs, metadata, verbose=False)

    # diag_report = DiagnosticReport()
    # diag_report.generate(train_df, correct_cfs, metadata)

    # shapes = qual_report.get_details(property_name='Column Shapes')

    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns

    # trends = qual_report.get_details(property_name='Column Pair Trends')
    # print(trends)


    quality =  qual_report.get_properties()
    Shape = quality['Score'][0] # Kolmogorov-Smirnov statistic for discrete, TVComplement for discrete
    Trend = quality['Score'][1] # CorrelationSimilarity for continuous-continuous, ContingencySimilarity for discrete-continuous, discrete-discrete


    if pyod:
        anomaly_score = np.mean(clf.predict_proba(correct_cfs, method="unify")[:, 0])
    else:
        anomaly_score = np.mean(clf.decision_function(correct_cfs)) # The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.

    return (Shape, Trend, anomaly_score)  


def gower_distance(x1, x2, cat_col_idx):
    is_categorical = np.isin(np.arange(len(x1)), cat_col_idx)

    is_numeric = ~is_categorical

    # categorical columns
    sij_cat = np.where(x1[is_categorical] == x2[is_categorical],np.zeros_like(x1[is_categorical]),np.ones_like(x1[is_categorical]))
    sum_cat = sij_cat.sum() 

    # numerical columns
    abs_delta=np.absolute(x1[is_numeric]-x2[is_numeric])
    sij_num=abs_delta # already normalized

    sum_num = sij_num.sum()
    sums= np.add(sum_cat,sum_num)
    feature_weight_sum = len(x1)
    sum_sij = np.divide(sums,feature_weight_sum)
    return sum_sij

def l0_distance(row1, row2):
    return (row1 != row2).sum()

def main(args):
    dataname = args.dataname
    method = args.method
    save_path = args.save_path
    num_samples = args.num_samples
    pyod = args.pyod
    
    # validity = args.validity
    # proximity = args.proximity
    # sparsity = args.sparsity
    # latent_clf = args.latent_clf

    dice_method = args.dice_method
    dice_optimization = args.dice_optimization
    proximity_weight_input = args.proximity_weight_input
    proximity_weight_latent = args.proximity_weight_latent
    proximity_latent_loss = args.proximity_latent_loss

    total_CFs = args.total_CFs
    get_stats = args.get_stats
    get_changed = args.get_changed


    optimization_parameters = get_optim_params_form_args(args) 

    data_dir = f'data/{dataname}'
    info_path = f'data/{dataname}/info.json'
    train_set_path = f'data/{dataname}/train.csv'




    output_folder_name = dataname


    if method == "dice":
        output_folder_name = f'{dataname}/DiCE/{dice_method}/{optimization_parameters}'
    elif method == "tabsyn":
        output_file_name_vae = get_name_form_args(args)
        output_folder_name = f'{dataname}/TABCF/{output_file_name_vae}/{dice_method}/{optimization_parameters}/'
    elif method == "revise":
        output_folder_name = f'{dataname}/Revise/'
    elif method == "cchvae":
        output_folder_name = f'{dataname}/CCHVAE/'
    elif method == "wachter":
        output_folder_name = f'{dataname}/Wachter/'

    if num_samples > 0:
        output_folder_name = f'{output_folder_name}/{num_samples}_samples'

    if total_CFs > 1:
        output_folder_name = f'{output_folder_name}_CFs_{total_CFs}'

    save_path = os.path.join(save_path, output_folder_name)

    # print("Evaluating:", output_folder_name)

    with open(info_path, 'r') as f:
        info = json.load(f)

    target_class = info["target_class"]




    cfs = pd.read_csv(os.path.join(save_path, "cfs.csv"), index_col=False)

    column_names = info["column_names"].copy()
    target_col = info["target_col"]
    column_names.remove(target_col)

    cfs = cfs[["id"] + column_names + ["target_prob"]]

    original_test_samples = pd.read_csv(os.path.join(save_path, "original_test_samples.csv"), index_col=False)



    # assert original_test_samples.iloc[:, info["target_col_idx"]].nunique()[0] == 1 or np.unique(original_test_samples[target_column])[0] != target_class, "Original test samples should all be from the negative class"


    up_arrow = "\u2191"
    down_arrow = "\u2193"

    cat_col_idx = info["cat_col_idx"]
    num_col_idx = info["num_col_idx"]
    num_cols = [column_names[x] for x in info["num_col_idx"]]
    cat_cols = [column_names[x] for x in info["cat_col_idx"]]
    num_max = {k: float(v) for k, v in info["num_max"].items()} 
    num_min = {k: float(v) for k, v in info["num_min"].items()}   

    train_df = pd.read_csv(train_set_path)
    # Initialize the StandardScaler
    scaler = StandardScaler()
    if len(num_cols)>0:
        # Fit the scaler on the numerical features
        scaler.fit(train_df[num_cols])

    del train_df

    valid_cfs = []
    metrics = []
    all_correct_cfs = []

    feature_changes_count = dict(zip(column_names, np.zeros((len(column_names)))))

    for indx, row in original_test_samples.iterrows():

        metrics_for_sample = []

        original_sample = original_test_samples.iloc[indx:indx+1]
        id = original_sample["id"].values[0]
        correct_cfs = cfs[(cfs["id"] == id) & (cfs["target_prob"] > 0.5)]


        if len(correct_cfs) == 0:
            continue

        valid_cfs.append(len(correct_cfs))

        correct_cfs_norm = correct_cfs.copy()
        correct_cfs_standard_norm = correct_cfs[num_cols].copy()
        original_sample_norm = original_sample.copy()

        original_sample_standard_norm = original_sample[num_cols].copy()

        if len(num_cols)>0:

            correct_cfs_norm[num_cols] = correct_cfs_norm[num_cols].apply(lambda x: (x - num_min[x.name]) / (num_max[x.name] - num_min[x.name]))
            original_sample_norm[num_cols] = original_sample_norm[num_cols].apply(lambda x: (x - num_min[x.name]) / (num_max[x.name] - num_min[x.name]))

            correct_cfs_standard_norm = scaler.transform(correct_cfs_standard_norm)
            original_sample_standard_norm = scaler.transform(original_sample_standard_norm)[0]

        correct_cfs_norm = correct_cfs_norm[column_names].reset_index(drop=True)
        original_sample_norm = original_sample_norm[column_names].to_numpy()[0]


        sparsity_for_sample = []
        sparsity_for_sample_cat = []
        sparsity_for_sample_cont = []
        proximity_for_sample = []
        proximity_for_sample_cont = []

        for c_i, _ in correct_cfs_norm.iterrows():
            cf = correct_cfs_norm.iloc[c_i:c_i+1].to_numpy()[0]
            if len(num_cols)>0:
                cf_standard = correct_cfs_standard_norm[c_i]

            for feature in column_names:
                if correct_cfs.iloc[c_i:c_i+1][feature].values != original_sample[feature].values:
                    feature_changes_count[feature] += 1

            sparsity_metric = l0_distance(cf, original_sample_norm)/len(cf)
            proximity_metric = gower_distance(cf, original_sample_norm, cat_col_idx)

            if len(num_cols)>0:

                proximity_cont = np.absolute(cf_standard, original_sample_standard_norm)

            sparsity_metric_cat = l0_distance(cf[cat_col_idx], original_sample_norm[cat_col_idx])/len(cat_col_idx)

            if len(num_cols)>0:

                sparsity_metric_cont = l0_distance(cf[num_col_idx], original_sample_norm[num_col_idx])/len(num_col_idx)


            sparsity_for_sample.append(sparsity_metric)
            sparsity_for_sample_cat.append(sparsity_metric_cat)
            if len(num_cols)>0:

                sparsity_for_sample_cont.append(sparsity_metric_cont)

            proximity_for_sample.append(proximity_metric)
            if len(num_cols)>0:
                proximity_for_sample_cont.append(proximity_cont)


            all_correct_cfs.append(correct_cfs.iloc[c_i:c_i+1])

        mean_sparsity = np.mean(sparsity_for_sample)    
        mean_sparsity_cat = np.mean(sparsity_for_sample_cat)   
        mean_proximity = np.mean(proximity_for_sample)    

        if len(num_cols)>0:
            mean_sparsity_cont = np.mean(sparsity_for_sample_cont)    
            mean_proximity_cont = np.mean(proximity_for_sample_cont)    

        if len(num_cols)>0:

            metrics.append([mean_sparsity, mean_proximity, mean_sparsity_cat, mean_sparsity_cont, mean_proximity_cont])
        else:
            metrics.append([mean_sparsity, mean_proximity, mean_sparsity_cat])

    average_metrics = np.mean(metrics, axis=0)

    if len(num_cols)>0:

        mean_df = pd.DataFrame([average_metrics], columns=[f'Sparsity {down_arrow}', f'Proximity (Gower) {down_arrow}', f'Sparsity cat {down_arrow}', f'Sparsity cont {down_arrow}', f'Prox cont {down_arrow}'])
    else:
        mean_df = pd.DataFrame([average_metrics], columns=[f'Sparsity {down_arrow}', f'Proximity (Gower) {down_arrow}', f'Sparsity cat {down_arrow}'])

    print(output_folder_name)

    len_cf_found = sum(valid_cfs)
    len_set = total_CFs * len(original_test_samples)

    mean_df[f"Validity black box {up_arrow}"] = len_cf_found/len_set

    mean_df.index = [method]

    train_set_path = f'data/{dataname}/train.csv'

    stats_df = pd.DataFrame()
    stats_df_target = pd.DataFrame()
    stats_df_negative = pd.DataFrame()
    outlier_df = pd.DataFrame()

    all_correct_cfs = pd.concat(all_correct_cfs)

    mean_df[f"Probability black box {up_arrow}"] = np.mean(all_correct_cfs["target_prob"])


    all_correct_cfs = all_correct_cfs.drop(["id", "target_prob"], axis=1)
    print(tabulate(mean_df, headers='keys', tablefmt='pretty'))

    mean_df.to_csv(os.path.join(save_path, "results.csv"))


    if get_stats:

        stats_all_data = get_plausibility_stats(all_correct_cfs, train_set_path, info, data_dir, pyod=pyod)
        stats_positive_data = get_plausibility_stats(all_correct_cfs, train_set_path, info, data_dir, target_only=True, pyod=pyod)
        stats_negative_data = get_plausibility_stats(all_correct_cfs, train_set_path, info, data_dir, negative_only=True, pyod=pyod)

        stats_df[f"Column-wise density (to train data) {up_arrow}"] = [stats_all_data[0]]
        stats_df[f"Pair-wise column correlation (to train data) {up_arrow}"] = [stats_all_data[1]]

        if pyod:
            outlier_df[f"Inlier probability (on train data) {up_arrow}"] = [stats_all_data[2]]
        else:
            outlier_df[f"Anomaly scores (on train data) {up_arrow}"] = [stats_all_data[2]]

        stats_df_target[f"Column-wise density (to positive class train data) {up_arrow}"] = [stats_positive_data[0]]
        stats_df_target[f"Pair-wise column correlation (to positive class train data) {up_arrow}"] = [stats_positive_data[1]]

        stats_df_negative[f"Column-wise density (to negative class train data) {up_arrow}"] = [stats_negative_data[0]]
        stats_df_negative[f"Pair-wise column correlation (to negative class train data) {up_arrow}"] = [stats_negative_data[1]]

        if pyod:
            outlier_df[f"Inlier probability (on positive class train data) {up_arrow}"] = [stats_positive_data[2]]
        else:
            outlier_df[f"Anomaly scores (on positive class train data) {up_arrow}"] = [stats_positive_data[2]]

        if pyod:
            outlier_df[f"Inlier probability (on negative class train data) {up_arrow}"] = [stats_negative_data[2]]
        else:
            outlier_df[f"Anomaly scores (on negative class train data) {up_arrow}"] = [stats_negative_data[2]]

        stats_df.index = [method]
        stats_df_target.index = [method]
        stats_df_negative.index = [method]
        outlier_df.index = [method]

        print(tabulate(stats_df, headers='keys', tablefmt='pretty'))
        print(tabulate(stats_df_target, headers='keys', tablefmt='pretty'))
        print(tabulate(stats_df_negative, headers='keys', tablefmt='pretty'))
        print(tabulate(outlier_df, headers='keys', tablefmt='pretty'))


    total_changes = sum(feature_changes_count.values())
    normalized_counter = {key: value / total_changes for key, value in feature_changes_count.items()}

    # Compute sums
    num_sum = sum(normalized_counter[col] for col in num_cols)
    cat_sum = sum(normalized_counter[col] for col in cat_cols)

    # Add new items to the dictionary
    normalized_counter['num_cols_sum'] = num_sum
    normalized_counter['cat_cols_sum'] = cat_sum
    if get_changed:
        print(normalized_counter)
        with open(os.path.join(save_path, "feature_changes.json"), 'w') as file:
            json.dump(normalized_counter, file, indent=4)  # 'indent=4' is optional, it makes the file more readable

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation Framework')

    args = parser.parse_args()
    args.device = 'cpu'
