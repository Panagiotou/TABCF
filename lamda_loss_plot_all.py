import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
dataname = "adult"
combine_sparsity = False

cmap_color_rev = sns.light_palette("seagreen", as_cmap=True, reverse=True)
cmap_color = sns.light_palette("seagreen", as_cmap=True)

proximity_weight_inputs = [0.0, 0.25, 0.5, 0.75, 1.0]
proximity_weight_latents = [0.0, 0.25, 0.5, 0.75, 1.0]
combinations = list(itertools.product(proximity_weight_inputs, proximity_weight_latents))
output_folder_name = f'counterfactual_results/{dataname}/DiCE_latent/num_encoding[min_max_torch]_num_loss[L2]_cat_actv[gumbel_softmax, tau=1.0]_cont_actv[sigmoid]_reparam/gradient'

# Initialize dictionaries to store results
sparsity_dict = {(i, l): np.nan for i, l in combinations}
sparsity_cat_dict = {(i, l): np.nan for i, l in combinations}
sparsity_cont_dict = {(i, l): np.nan for i, l in combinations}
proximity_dict = {(i, l): np.nan for i, l in combinations}
validity_dict = {(i, l): np.nan for i, l in combinations}
confidence_dict = {(i, l): np.nan for i, l in combinations}

# Read data for each combination
for proximity_weight_input, proximity_weight_latent in combinations:
    experiment_folder = f'proximity_weight_input[{proximity_weight_input}]_hidden_dims[16]'
    if proximity_weight_latent > 0:
        experiment_folder += f'_proximity_weight_latent[{proximity_weight_latent}]_proximity_latent_loss[L2]'
    
    result_file = f'{output_folder_name}/{experiment_folder}/1000_samples/results.csv'
    
    try:
        res = pd.read_csv(result_file)
        if not combine_sparsity:
            sparsity = float(res["Sparsity cat ↓"].values[0])
        else:
            sparsity = (float(res["Sparsity cat ↓"].values[0]) + float(res["Sparsity cont ↓"].values[0]))/2

        proximity = float(res["Prox cont ↓"].values[0])
        validity = float(res["Validity black box ↑"].values[0])
        confidence = float(res["Probability black box ↑"].values[0])

        sparsity_cat = float(res["Sparsity cat ↓"].values[0])
        sparsity_cont = float(res["Sparsity cont ↓"].values[0])

        
        # Store values in dictionaries
        sparsity_dict[(proximity_weight_input, proximity_weight_latent)] = sparsity
        sparsity_cat_dict[(proximity_weight_input, proximity_weight_latent)] = sparsity_cat
        sparsity_cont_dict[(proximity_weight_input, proximity_weight_latent)] = sparsity_cont
        proximity_dict[(proximity_weight_input, proximity_weight_latent)] = proximity
        validity_dict[(proximity_weight_input, proximity_weight_latent)] = validity
        confidence_dict[(proximity_weight_input, proximity_weight_latent)] = confidence
    except FileNotFoundError:
        print(f"File {result_file} not found. Skipping this combination.")
    except Exception as e:
        print(f"An error occurred while reading {result_file}: {e}. Skipping this combination.")

# Convert dictionaries to DataFrames
sparsity_df = pd.DataFrame(list(sparsity_dict.items()), columns=['Index', 'Sparsity'])

sparsity_cat_df = pd.DataFrame(list(sparsity_cat_dict.items()), columns=['Index', 'sparsity_cat'])
sparsity_cont_df = pd.DataFrame(list(sparsity_cont_dict.items()), columns=['Index', 'sparsity_cont'])

proximity_df = pd.DataFrame(list(proximity_dict.items()), columns=['Index', 'Proximity'])
validity_df = pd.DataFrame(list(validity_dict.items()), columns=['Index', 'Validity'])
confidence_df = pd.DataFrame(list(confidence_dict.items()), columns=['Index', 'Confidence'])

# Split the tuple in the 'Index' column into separate columns
sparsity_df[['Proximity Weight Input', 'Proximity Weight Latent']] = pd.DataFrame(sparsity_df['Index'].tolist(), index=sparsity_df.index)
proximity_df[['Proximity Weight Input', 'Proximity Weight Latent']] = pd.DataFrame(proximity_df['Index'].tolist(), index=proximity_df.index)
validity_df[['Proximity Weight Input', 'Proximity Weight Latent']] = pd.DataFrame(validity_df['Index'].tolist(), index=validity_df.index)
confidence_df[['Proximity Weight Input', 'Proximity Weight Latent']] = pd.DataFrame(confidence_df['Index'].tolist(), index=confidence_df.index)

sparsity_cat_df[['Proximity Weight Input', 'Proximity Weight Latent']] = pd.DataFrame(sparsity_cat_df['Index'].tolist(), index=sparsity_cat_df.index)
sparsity_cont_df[['Proximity Weight Input', 'Proximity Weight Latent']] = pd.DataFrame(sparsity_cont_df['Index'].tolist(), index=sparsity_cont_df.index)

# Drop the original 'Index' column
sparsity_df = sparsity_df.drop(columns='Index')
proximity_df = proximity_df.drop(columns='Index')
validity_df = validity_df.drop(columns='Index')
confidence_df = confidence_df.drop(columns='Index')

sparsity_cat_df = sparsity_cat_df.drop(columns='Index')
sparsity_cont_df = sparsity_cont_df.drop(columns='Index')


# Pivot the DataFrames to have the correct format for heatmaps
sparsity_pivot = sparsity_df.pivot(index='Proximity Weight Input', columns='Proximity Weight Latent', values='Sparsity')
proximity_pivot = proximity_df.pivot(index='Proximity Weight Input', columns='Proximity Weight Latent', values='Proximity')
validity_pivot = validity_df.pivot(index='Proximity Weight Input', columns='Proximity Weight Latent', values='Validity')
confidence_pivot = confidence_df.pivot(index='Proximity Weight Input', columns='Proximity Weight Latent', values='Confidence')


sparsity_cat_pivot = sparsity_cat_df.pivot(index='Proximity Weight Input', columns='Proximity Weight Latent', values='sparsity_cat')
sparsity_cont_pivot = sparsity_cont_df.pivot(index='Proximity Weight Input', columns='Proximity Weight Latent', values='sparsity_cont')

# Reverse the order of rows (y-axis) to have smallest values at the bottom
sparsity_pivot = sparsity_pivot.sort_index(ascending=False)
proximity_pivot = proximity_pivot.sort_index(ascending=False)
validity_pivot = validity_pivot.sort_index(ascending=False)
confidence_pivot = confidence_pivot.sort_index(ascending=False)

sparsity_cat_pivot = sparsity_cat_pivot.sort_index(ascending=False)
sparsity_cont_pivot = sparsity_cont_pivot.sort_index(ascending=False)

# Sort columns (x-axis) to ensure correct ordering
sparsity_pivot = sparsity_pivot[sorted(sparsity_pivot.columns)]
proximity_pivot = proximity_pivot[sorted(proximity_pivot.columns)]
validity_pivot = validity_pivot[sorted(validity_pivot.columns)]
confidence_pivot = confidence_pivot[sorted(confidence_pivot.columns)]

sparsity_cat_pivot = sparsity_cat_pivot[sorted(sparsity_cat_pivot.columns)]
sparsity_cont_pivot = sparsity_cont_pivot[sorted(sparsity_cont_pivot.columns)]

# Plotting
fig, ax = plt.subplots(2, 2, figsize=(20,18))

top_font = 35
label_font = 35
tick_font = 25
colorbar_font = 16






# Heatmap for Validity
heatmap_validity = sns.heatmap(validity_pivot, ax=ax[0][0], cmap=cmap_color, annot=True, annot_kws={'size': tick_font})
ax[0][0].set_title('Validity ↑', fontsize=top_font, pad=20)
ax[0][0].set_xlabel(r'$\lambda_{prox\_latent}$', fontsize=label_font, labelpad=15)
ax[0][0].set_ylabel(r'$\lambda_{prox\_input}$', fontsize=label_font, labelpad=15)
ax[0][0].tick_params(axis='both', which='major', labelsize=tick_font)
colorbar_validity = heatmap_validity.collections[0].colorbar
colorbar_validity.ax.tick_params(labelsize=colorbar_font)

# Heatmap for Proximity
heatmap_proximity = sns.heatmap(proximity_pivot, ax=ax[0][1], cmap=cmap_color_rev, annot=True, annot_kws={'size': tick_font})
ax[0][1].set_title('Proximity Num↓', fontsize=top_font, pad=20)
ax[0][1].set_xlabel(r'$\lambda_{prox\_latent}$', fontsize=label_font, labelpad=15)
ax[0][1].tick_params(axis='both', which='major', labelsize=tick_font)
colorbar_proximity = heatmap_proximity.collections[0].colorbar
colorbar_proximity.ax.tick_params(labelsize=colorbar_font)



# Heatmap for Sparsity
heatmap_sparsity = sns.heatmap(sparsity_cont_pivot, ax=ax[1][0], cmap=cmap_color_rev, annot=True, annot_kws={'size': tick_font})
ax[1][0].set_title('Sparsity Num↓', fontsize=top_font, pad=20)
ax[1][0].set_xlabel(r'$\lambda_{prox\_latent}$', fontsize=label_font, labelpad=15)
ax[1][0].set_ylabel(r'$\lambda_{prox\_input}$', fontsize=label_font, labelpad=15)
ax[1][0].tick_params(axis='both', which='major', labelsize=tick_font)
colorbar_sparsity = heatmap_sparsity.collections[0].colorbar
colorbar_sparsity.ax.tick_params(labelsize=colorbar_font)

# Heatmap for Sparsity
heatmap_sparsity = sns.heatmap(sparsity_cat_pivot, ax=ax[1][1], cmap=cmap_color_rev, annot=True, annot_kws={'size': tick_font})
ax[1][1].set_title('Sparsity Cat↓', fontsize=top_font, pad=20)
ax[1][1].set_xlabel(r'$\lambda_{prox\_latent}$', fontsize=label_font, labelpad=15)
ax[1][1].set_ylabel(r'$\lambda_{prox\_input}$', fontsize=label_font, labelpad=15)
ax[1][1].tick_params(axis='both', which='major', labelsize=tick_font)
colorbar_sparsity = heatmap_sparsity.collections[0].colorbar
colorbar_sparsity.ax.tick_params(labelsize=colorbar_font)



# # Heatmap for coonfidence
# heatmap_confidence = sns.heatmap(confidence_pivot, ax=ax[1][1], cmap=cmap_color, annot=True, annot_kws={'size': tick_font})
# ax[1][1].set_title('Confidence ↑', fontsize=top_font, pad=20)
# ax[1][1].set_xlabel(r'$\lambda_{prox\_latent}$', fontsize=label_font, labelpad=15)
# ax[1][1].tick_params(axis='both', which='major', labelsize=tick_font)
# colorbar_confidence = heatmap_confidence.collections[0].colorbar
# colorbar_confidence.ax.tick_params(labelsize=colorbar_font)


# Remove y-axis labels from the right plots
ax[0][0].set_xlabel('')
ax[0][1].set_xlabel('')
ax[0][1].set_ylabel('')
ax[1][1].set_ylabel('')

# # Remove x-axis labels from the top plots
# ax[0].set_xlabel('')
# ax[2].set_xlabel('')

# Adjust layout to make room for titles and labels
plt.subplots_adjust(wspace=0.1)

# Save the figure
if combine_sparsity:
    plt.savefig(f'proximity_weight_effects_heatmap_{dataname}_all_new_combine_sparsity.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(f'proximity_weight_effects_heatmap_{dataname}_all_new.png', format='png', dpi=300, bbox_inches='tight')
print("saved", f'proximity_weight_effects_heatmap_{dataname}_all_new.png' )
# plt.show()
