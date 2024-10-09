from utils_train import preprocess, TabularDataset, get_name_form_args, get_optim_params_form_args

from tabsyn.vae.model import BBMLPCLF
import torch
import shap
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import json 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

dataname = "adult"
data_dir = f'data/{dataname}'
hidden_dims = 16

gradient = False

percentage = False

num_samples = 1000
target = 1

device = 'cuda'

test_set_path = f'data/{dataname}/test.csv'
train_set_path = f'data/{dataname}/train.csv'

with open(f'{data_dir}/info.json', 'r') as f:
    info = json.load(f)

train_df = pd.read_csv(train_set_path)
test_df = pd.read_csv(test_set_path)

num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

all_cols = num_cols + cat_cols
test_df = test_df[all_cols]
train_df = train_df[all_cols]

# Create preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ]
)
preprocessor.fit(train_df)

X_num_cat_ohe = preprocessor.transform(test_df)

model_filename = f'{data_dir}/black_box_mlp.pkl'

if hidden_dims is not None:
    model_filename = f'{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl'

input_shape = X_num_cat_ohe[0].shape[-1]

loaded_black_box_clf = BBMLPCLF(input_shape).to(device)
loaded_black_box_clf.load_state_dict(torch.load(model_filename))

cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

f = lambda x: loaded_black_box_clf(Variable(torch.from_numpy(np.array(x)).float().to(device))).detach().cpu().numpy()[:, target]

test_encoded = X_num_cat_ohe
test_pred = f(test_encoded)
negative_indices = np.where(test_pred < 0.5)[0]

negative_test_encoded = test_encoded[negative_indices][:num_samples]
negative_df = test_df.iloc[negative_indices][:num_samples]

test_encoded = negative_test_encoded
test_df = negative_df

# Convert the result to a DataFrame
preprocessed_df = pd.DataFrame(test_encoded, columns=all_feature_names)

if gradient:
    explainer = shap.GradientExplainer(loaded_black_box_clf, torch.tensor(test_encoded).to(device).float())
    shap_values = explainer(torch.tensor(test_encoded).to(device).float())
else:
    explainer = shap.Explainer(f, preprocessed_df)
    shap_values = explainer(preprocessed_df)

summed_shap_values = {col: 0 for col in num_cols + cat_cols}

# Get the SHAP values for each feature
shap_values_array = shap_values.values

if gradient:
    shap_values_array = shap_values.values[:, :, target]
else:
    shap_values_array = shap_values.values

# Sum SHAP values for numerical features (they don't need summing actually, just copying)
for col in num_cols:
    summed_shap_values[col] = shap_values_array[:, preprocessed_df.columns.get_loc(col)]

# Sum SHAP values for one-hot encoded categorical features
for col in cat_cols:
    one_hot_encoded_feature_names = preprocessed_df.columns[preprocessed_df.columns.str.startswith(col + '_')]
    summed_shap_values[col] = shap_values_array[:, [preprocessed_df.columns.get_loc(name) for name in one_hot_encoded_feature_names]].sum(axis=1)

# Convert the result to a DataFrame
summed_shap_df = pd.DataFrame(summed_shap_values)

if percentage:
    feature_importance = summed_shap_df.abs().sum(axis=0)
    feature_importance_normalized = feature_importance / feature_importance.sum()
    feature_importance_df = pd.DataFrame({
        'feature': feature_importance_normalized.index,
        'importance': feature_importance_normalized.values
    }).sort_values(by='importance', ascending=False)

    feature_importance_percentage = shap.Explanation(
        values=feature_importance_df['importance'].values,
        feature_names=feature_importance_df['feature'].tolist()
    )

    dict_fi = feature_importance_normalized.to_dict()
    num_sum = sum(dict_fi[col] for col in num_cols)
    cat_sum = sum(dict_fi[col] for col in cat_cols)

    dict_fi['num_cols_sum'] = num_sum
    dict_fi['cat_cols_sum'] = cat_sum

    print(dict_fi)
    shap.plots.bar(feature_importance_percentage, max_display=len(num_cols) + len(cat_cols), show=False)
    plt.savefig(f'standard_shap_{dataname}.png', dpi=100, bbox_inches='tight')
else:
    summed_shap_explanation = shap.Explanation(
        values=summed_shap_df.values,
        feature_names=summed_shap_df.columns.tolist(),
        data=np.array(test_df)
    )

    shap.plots.beeswarm(summed_shap_explanation, max_display=len(num_cols) + len(cat_cols), show=False, color=plt.get_cmap("cool"))

    # Customize y-axis labels to make numerical feature names bold
    ax = plt.gca()
    ytick_labels = ax.get_yticklabels()
    for label in ytick_labels:
        if label.get_text() in num_cols:
            label.set_fontweight('bold')


    # Rename the y-axis label "hours.per.week" to "hours/week"
    ytick_labels = [label.get_text().replace('hours.per.week', 'hours/week') for label in ytick_labels]

    ytick_labels = [label.replace('marital.status', 'marital\nstatus') for label in ytick_labels]
    ytick_labels = [label.replace('native.country', 'native\ncountry') for label in ytick_labels]
    ax.set_yticklabels(ytick_labels)

    # Define start and end points for the arrow
    start = (0.6, 0.05)
    end = (0.97, 0.05)

    # Draw the arrow using annotate
    ax.annotate('', xy=end, xycoords='axes fraction',
                xytext=start, textcoords='axes fraction',
                arrowprops=dict(facecolor='black', edgecolor='black',
                                arrowstyle='->', linewidth=2))

    # Compute the position for the text
    text_position = (0.3, start[1] + (end[1] - start[1]) / 2)


    fig, ax = plt.gcf(), plt.gca()

    # Place text manually at the computed position
    ax.text(text_position[0], text_position[1], 'target class',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='black', weight='bold')

    ax.text(text_position[0], text_position[1]-0.75, '(positive impact)',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='black', weight='bold')





    plt.savefig(f'standard_shap_{dataname}_raw.png', format='png', dpi=300, bbox_inches='tight')

