import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Configuration to use LaTeX in Matplotlib
plt.rc('text', usetex=True)

# Define font sizes for different plot elements
title_fontsize = 20
xlabel_fontsize = 15
ylabel_fontsize = 40
tick_params_fontsize_x = 15  # Font size for x-axis ticks
tick_params_fontsize_y = 15  # Font size for y-axis ticks
bar_label_fontsize = 15
legend_fontsize = 15

dataname = "adult"
output_folder_name = f'counterfactual_results/{dataname}'

methods = ["Wachter", "DiCE", "Revise", "CCHVAE", "DiCE_latent"]

us = r'\textbf{T\Large{\textbf{AB}}\LARGE{C}\Large{\textbf{F}}}'
# Simplified and corrected LaTeX formatted names
names = [
    r'\textbf{Wachter}', 
    r'\textbf{DiCE}', 
    r'\textbf{REVISE}', 
    r'\textbf{CCHVAE}', 
    us
]

columns = ["marital.status", "education", "occupation", "age", "relationship", "sex", "hours.per.week", "workclass", "capital.gain", "capital.loss", "race", "native.country"]

# List of numerical columns
numerical_columns = ["age", "hours.per.week", "capital.gain", "capital.loss"]

columns.reverse()

fig, axs = plt.subplots(1, len(methods), figsize=(4 * len(methods), 6), sharey=True)

if len(methods) == 1:
    axs = [axs]

for i, method in enumerate(methods):
    output_folder_name_method = f'{output_folder_name}/{method}'

    if method == "DiCE_latent":
        output_folder_name_method = f'{output_folder_name_method}/num_encoding[min_max_torch]_num_loss[L2]_cat_actv[gumbel_softmax, tau=1.0]_cont_actv[sigmoid]_reparam/gradient/proximity_weight_input[1.0]_hidden_dims[16]_proximity_weight_latent[1.0]_proximity_latent_loss[L2]'
    if method == "DiCE":
        output_folder_name_method = f'{output_folder_name_method}/gradient/proximity_weight_input[1.0]_hidden_dims[16]'

    feature_c_json = f'{output_folder_name_method}/1000_samples/feature_changes.json'
    with open(feature_c_json, 'r') as file:
        data = pd.DataFrame.from_dict([json.load(file)])

    # Extract values for plotting
    values = data[columns].iloc[0]

    # Ensure values is a Series or 1D array
    if values.ndim > 1:
        values = values.squeeze()

    ax = axs[i]
    colors = ['royalblue' if col in numerical_columns else 'grey' for col in columns]
    bars = ax.barh(columns, values, color=colors)
    ax.set_title(names[i], fontsize=title_fontsize)

    # Calculate the total percentage for numerical and categorical features
    total_numerical = sum(values[col] for col in numerical_columns)
    total_categorical = sum(values[col] for col in columns if col not in numerical_columns)

    # Annotate each bar with its value inside the bar
    for bar in bars:
        width = bar.get_width()
        if width > 0.03:
            ax.text(
                width - (width * 0.02),  # Position the text slightly left from the end of the bar
                bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}',  # Format the value with two decimal places
                va='center',
                ha='right',
                fontsize=bar_label_fontsize,
                color='white'
            )

    # Create legend patches
    legend_patches = [
        Patch(color='royalblue', label=f'Numerical: {total_numerical:.2f}%'),
        Patch(color='grey', label=f'Categorical: {total_categorical:.2f}%')
    ]

    # Add legend with total percentages for numerical and categorical features
    legend = ax.legend(
        handles=legend_patches,
        title='',
        loc='lower right',
        fontsize=legend_fontsize,
        frameon=True,  # Add box around the legend
        title_fontsize=legend_fontsize
    )

    # Customize tick parameters
    ax.tick_params(axis='x', which='major', labelsize=tick_params_fontsize_x)
    ax.tick_params(axis='y', which='major', labelsize=tick_params_fontsize_y)

    # Customize y-tick labels for numerical columns to be bold
    yticks = ax.get_yticks()
    yticklabels = [col if col not in numerical_columns else rf'\textbf{{{col}}}' for col in columns]

    yticklabels = [label.replace('hours.per.week', 'hours/week') for label in yticklabels]
    yticklabels = [label.replace('marital.status', 'marital\nstatus') for label in yticklabels]
    yticklabels = [label.replace('native.country', 'native\ncountry') for label in yticklabels]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=tick_params_fontsize_y)

# Add an overall x-axis label
fig.text(0.5, 0.04, 'Feature utilization (\%)', ha='center', va='center', fontsize=title_fontsize)

plt.savefig(f'feature_changes_adult.png', format='png', dpi=300, bbox_inches='tight')
