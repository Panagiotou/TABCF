"""
Module to generate diverse counterfactual explanations based on PyTorch framework
"""
import copy
import random
import timeit

import numpy as np
import torch

from dice_ml import diverse_counterfactuals as exp
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class DicePyTorch(ExplainerBase):

    def __init__(self, data_interface, model_interface, data_dice_input=None, dice_model_on_input=None, verbose=False, loaded_gmm=None, train_z_df_target=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        # initiating data related parameters
        super().__init__(data_interface)
        # initializing model related variables
        self.model = model_interface
        self.model.load_model()  # loading trained model
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()
        # temp data to create some attributes like encoded feature names
        temp_ohe_data = self.model.transformer.transform(self.data_interface.data_df.iloc[[0]])
        self.data_interface.create_ohe_params(temp_ohe_data)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()
        

        self.data_dice_input = data_dice_input
        self.dice_model_on_input = dice_model_on_input
        self.device = model_interface.device

        self.loaded_gmm = loaded_gmm

        if train_z_df_target is not None:
            train_z_df_target = torch.Tensor(train_z_df_target.values).to(self.device)

            self.train_z_df_target = train_z_df_target

            self.maha_mean, self.maha_inv_cov_matrix = self.precompute_mahalanobis_params(train_z_df_target)

        if self.data_dice_input is not None:
            self.decoded_minx, self.decoded_maxx, self.decoded_encoded_categorical_feature_indexes, self.decoded_encoded_continuous_feature_indexes, \
                self.decoded_cont_minx, self.decoded_cont_maxx, self.decoded_cont_precisions = self.data_dice_input.get_data_params_for_gradient_dice()
            

        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.ohe_encoded_feature_names)).shape[1]

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.cfs_input = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.verbose = verbose

    def precompute_mahalanobis_params(self, group_instances):
        # Compute the mean and covariance matrix of the group instances
        mean = group_instances.mean(0)  # Compute mean across the first dimension
        cov_matrix = torch.cov(group_instances.T)  # Compute covariance matrix
        inv_cov_matrix = torch.inverse(cov_matrix)  # Compute inverse covariance matrix
        return mean, inv_cov_matrix

    def _generate_counterfactuals(self, query_instance, total_CFs,
                                  desired_class="opposite", desired_range=None,
                                  proximity_weight=0.5,
                                  diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all",
                                  permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", optimizer="pytorch:adam", learning_rate=0.05, min_iter=500,
                                  max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                  init_near_query_instance=True, tie_random=False, stopping_threshold=0.5, proximity_weight_latent=-1.0, proximity_latent_loss="L2", proximity_input_loss=None, plausibility_weight_latent=-1.0, plausibility_latent_loss="GMM", latent=False,
                                  posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", limit_steps_ls=10000, decode_before_loss=False, original_input_instance=None, plot_gradients=False, immutable_mask=[], index=0):
        """Generates diverse counterfactual explanations.

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class
                              of query_instance for binary classification.
        :param desired_range: Not supported currently.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                               Defaults to the range inferred from training data. If None, uses the parameters initialized in
                               data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as
                                values. Default option is "inverse_mad" where the weight for a continuous feature is the
                                inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set;
                                the weight for a categorical feature is equal to 1 by default.
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".
        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence.
                                      Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement

        :return: A CounterfactualExamples object to store and visualize the resulting
                 counterfactual explanations (see diverse_counterfactuals.py).
        """
        self.decode_before_loss = decode_before_loss

        
        self.latent = latent
        self.plot_gradients = plot_gradients

        self.immutable_mask = immutable_mask

        self.index = index

        if len(immutable_mask)>0:
            self.immutable_mask = torch.tensor(immutable_mask).to(self.device)

        if original_input_instance is not None:
            self.original_input_instance = original_input_instance
            self.test_instance_df = self.model.transformer.inverse_transform(self.data_dice_input.get_decoded_data(original_input_instance))
            

        if not self.decode_before_loss:
            # check feature MAD validity and throw warnings
            if feature_weights == "inverse_mad":
                self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        if [total_CFs, algorithm, features_to_vary] != self.cf_init_weights:
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if [yloss_type, diversity_loss_type, feature_weights] != self.loss_weights:
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if [proximity_weight, diversity_weight, categorical_penalty] != self.hyperparameters:
            self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

        self.proximity_weight_latent = proximity_weight_latent
        self.proximity_latent_loss = proximity_latent_loss

        self.plausibility_latent_loss = plausibility_latent_loss
        self.proximity_input_loss = proximity_input_loss
        self.plausibility_weight_latent = plausibility_weight_latent

        final_cfs_df, test_instance_df, final_cfs_df_sparse = \
            self.find_counterfactuals(
                query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter,
                project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance,
                tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, limit_steps_ls)

        return exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class)

    def get_model_output(self, input_instance,
                         transform_data=False, out_tensor=True):
        """get output probability of ML model"""
        return self.model.get_output(
                input_instance,
                transform_data=transform_data,
                out_tensor=out_tensor)[(self.num_output_nodes-1):]
    
    def get_model_output_bb_only(self, input_instance,
                         transform_data=False, out_tensor=True):
        """get output probability of ML model"""
        return self.model.get_output_bb_only(
                input_instance,
                transform_data=transform_data,
                out_tensor=out_tensor)[(self.num_output_nodes-1):]

    def predict_fn(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()
        return self.get_model_output(
                input_instance, transform_data=False, out_tensor=False)

    def predict_fn_bb_only(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()
        return self.get_model_output_bb_only(
                input_instance, transform_data=False, out_tensor=False)
    
    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        input_instance = self.model.transformer.transform(input_instance).to_numpy()[0]
        return self.predict_fn(torch.tensor(input_instance).float())

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
            self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)

        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            for ix in range(self.total_CFs):
                one_init = []
                self.cfs_input.append([])

                for jx in range(self.minx.shape[1]):
                    one_init.append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                self.cfs.append(torch.tensor(one_init).float().to(self.device))
                self.cfs[ix].requires_grad = True

    def get_feature_weights(self, feature_weights="inverse_mad"):
        feature_weights_input = feature_weights
        if feature_weights == "inverse_mad":
            normalized_mads = self.data_interface.get_valid_mads(normalized=True)
            feature_weights = {}
            for feature in normalized_mads:
                feature_weights[feature] = round(1/normalized_mads[feature], 2)

        feature_weights_list = []
        for feature in self.data_interface.ohe_encoded_feature_names:
            if feature in feature_weights:
                feature_weights_list.append(feature_weights[feature])
            else:
                feature_weights_list.append(1.0)
        feature_weights_list = torch.tensor(feature_weights_list)

        return feature_weights_list


    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        if self.decode_before_loss:
            self.feature_weights_list = feature_weights.to(self.device)
        else:
            # define feature weights
            if feature_weights != self.feature_weights_input:
                self.feature_weights_input = feature_weights
                if feature_weights == "inverse_mad":
                    normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                    feature_weights = {}
                    for feature in normalized_mads:
                        feature_weights[feature] = round(1/normalized_mads[feature], 2)

                feature_weights_list = []
                for feature in self.data_interface.ohe_encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
                self.feature_weights_list = torch.tensor(feature_weights_list).to(self.device)


        # define different parts of loss function
        self.yloss_opt = torch.nn.BCEWithLogitsLoss()

    def update_hyperparameters(self, proximity_weight, diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based PyTorch optimizers."""
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(self.cfs, lr=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.cfs, lr=learning_rate)

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        for i in range(self.total_CFs):
            if self.yloss_type == "l2_loss":
                temp_loss = torch.pow((self.get_model_output(self.cfs[i]) - self.target_cf_class), 2)[0]
            elif self.yloss_type == "log_loss":
                temp_logits = torch.log(
                    (abs(self.get_model_output(self.cfs[i]) - 0.000001)) /
                    (1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.BCEWithLogitsLoss()
                temp_loss = criterion(temp_logits, torch.tensor([self.target_cf_class]))
            elif self.yloss_type == "hinge_loss":
                temp_logits = torch.log(
                    (abs(self.get_model_output(self.cfs[i]) - 0.000001)) /
                    (1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.ReLU()
                all_ones = torch.ones_like(self.target_cf_class)
                labels = 2 * self.target_cf_class - all_ones
                temp_loss = all_ones - torch.mul(labels, temp_logits)
                temp_loss = torch.norm(criterion(temp_loss))

            yloss += temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        # print(torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0))
        # print(x_hat)
        # print(x1)
        # print(self.feature_weights_list)
        # print(torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0))
        # exit(1)
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0)

    def compute_dist_latent(self, x_hat, x1, dist="L2"):
        """Compute weighted distance between two vectors."""
        # print(torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0))
        # print(x_hat)
        # print(x1)
        # print(self.feature_weights_list)
        # print(torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0))
        # exit(1)
        if dist=="L2":
            return torch.sum((x_hat - x1) ** 2, dim=0)
        else:
            return torch.sum(torch.abs(x_hat - x1), dim=0)

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            if self.decode_before_loss:
                decoded_cf_i = self.model.get_decoded(self.cfs[i])
                if self.proximity_input_loss=="reconstruction":
                    cat_loss = 0
                    l1_loss_fn = torch.nn.L1Loss()

                    x_hat_n = decoded_cf_i[self.decoded_encoded_continuous_feature_indexes]
                    x1_n = self.x1_decoded[self.decoded_encoded_continuous_feature_indexes]

                    num_loss = torch.sum(torch.mul((torch.abs(x_hat_n - x1_n)), self.feature_weights_list[self.decoded_encoded_continuous_feature_indexes]), dim=0)/len(self.decoded_encoded_continuous_feature_indexes)

                    for v in self.decoded_encoded_categorical_feature_indexes:
                        cat_loss += l1_loss_fn(decoded_cf_i[v[0]:v[-1]+1], self.x1_decoded[v[0]:v[-1]+1])
                        
                    cat_loss /= len(self.decoded_encoded_categorical_feature_indexes)
                    proximity_loss += num_loss + cat_loss
                else:
                    proximity_loss += self.compute_dist(decoded_cf_i, self.x1_decoded)
            else:
                proximity_loss += self.compute_dist(self.cfs[i], self.x1)

        if self.decode_before_loss:
            if self.proximity_input_loss=="reconstruction":
                return proximity_loss/(self.total_CFs)
            
            return proximity_loss/(torch.mul(len(self.decoded_minx[0]), self.total_CFs))
        else:
            return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs))

    def compute_proximity_loss_latent(self):
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist_latent(self.cfs[i], self.x1, dist=self.proximity_latent_loss)
        return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs))


    def compute_mahalanobis_distance(self, instance):
        diff = instance - self.maha_mean
        distance = torch.sqrt(torch.matmul(torch.matmul(diff, self.maha_inv_cov_matrix), diff.T))
        return distance


    def knn_loss(self, x, k=3):
        """
        Computes the k-nearest neighbors loss for a query instance.
        :param query_instance: Query instance to optimize, shape (1, D).
        :param data: Dataset of instances, shape (N, D).
        :param k: Number of nearest neighbors to consider.
        :return: Loss value.
        """

        x = x.unsqueeze(0)  # shape (1, 48)
        # Compute Euclidean distances
        dists = torch.sum((self.train_z_df_target - x)**2, dim=1)
        
        # Find indices of k smallest distances
        _, indices = torch.topk(dists, k, largest=False)
        
        # Calculate mean squared distance to nearest neighbors
        mean_dist = torch.mean(torch.sum((self.train_z_df_target[indices] - x)**2, dim=1))
        
        return mean_dist

    def compute_plausibility_loss_latent(self):
        plausibility_loss = 0.0
        for i in range(self.total_CFs):
            if self.plausibility_latent_loss == "GMM":
                plausibility_loss += self.loaded_gmm.forward(self.cfs[i]) 
            elif self.plausibility_latent_loss == "mahalanobis":
                plausibility_loss += self.compute_mahalanobis_distance(self.cfs[i])
            elif self.plausibility_latent_loss == "KDE":
                plausibility_loss += self.density_loss(self.cfs[i])
            elif self.plausibility_latent_loss == "KNN":
                plausibility_loss += self.knn_loss(self.cfs[i])
            else:
                print("Plausibility method not implemented:", self.plausibility_latent_loss)
                exit(1)

        return plausibility_loss/self.total_CFs
    

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    if self.latent:
                        det_entries[(i, j)] = 1.0/(1.0 + self.compute_dist_latent(self.cfs[i], self.cfs[j]))
                    else:
                        det_entries[(i, j)] = 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    if self.latent:
                        det_entries[(i, j)] = 1.0/(torch.exp(self.compute_dist_latent(self.cfs[i], self.cfs[j])))
                    else:
                        det_entries[(i, j)] = 1.0/(torch.exp(self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions -
           to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += torch.pow((torch.sum(self.cfs[i][v[0]:v[-1]+1]) - 1.0), 2)

        return regularization_loss

    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.proximity_loss_latent = self.compute_proximity_loss_latent() if self.proximity_weight_latent > 0 else 0.0
        self.plausibility_loss_latent = self.compute_plausibility_loss_latent() if self.plausibility_weight_latent > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        
        self.regularization_loss = self.compute_regularization_loss()


        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) + (self.proximity_weight_latent * self.proximity_loss_latent) + (self.plausibility_weight_latent * self.plausibility_loss_latent) - \
            (self.diversity_weight * self.diversity_loss) + \
            (self.categorical_penalty * self.regularization_loss)

        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        self.cfs[n].data[i] = query_instance[i]+(n*0.01)
                    else:
                        self.cfs[n].data[i] = np.random.uniform(self.minx[0][i], self.maxx[0][i])
                else:
                    self.cfs[n].data[i] = query_instance[i]

    def decoded_round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = self.model.get_decoded(tcf).detach().clone().cpu().numpy()

            for i, v in enumerate(self.decoded_encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[v]*(self.decoded_cont_maxx[i] - self.decoded_cont_minx[i])) + self.decoded_cont_minx[i]
                org_cont = round(org_cont, self.decoded_cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.decoded_cont_minx[i])/(self.decoded_cont_maxx[i] - self.decoded_cont_minx[i])
                cf[v] = normalized_cont  # assign the projected continuous value

            for v in self.decoded_encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[v[vi]] = 1.0
                    else:
                        cf[v[vi]] = 0.0


            cf = torch.tensor(cf).to(self.device)

            self.cfs_input[index] = cf


            if assign:
                cf_encoded = self.model.get_encoded(cf)
                temp_cfs.append(cf_encoded)
                for jx in range(len(cf_encoded)):
                    self.cfs[index].data[jx] = temp_cfs[index][jx]
            else:
                temp_cfs.append(cf)

        if assign:
            return None
        else:
            return temp_cfs
        
    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.detach().clone().cpu().numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                org_cont = round(org_cont, self.cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[v] = normalized_cont  # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[v[vi]] = 1.0
                    else:
                        cf[v[vi]] = 0.0

            cf = torch.tensor(cf).to(self.device)

            temp_cfs.append(cf)
            if assign:
                for jx in range(len(cf)):
                    self.cfs[index].data[jx] = temp_cfs[index][jx]

        if assign:
            return None
        else:
            return temp_cfs

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if self.project_iter > 0 and itr > 0:
            if itr % self.project_iter == 0:
                if self.decode_before_loss:
                    self.decoded_round_off_cfs(assign=True)
                else:
                    self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                if self.decode_before_loss:
                    temp_cfs = self.decoded_round_off_cfs(assign=False)
                    test_preds = [self.predict_fn_bb_only(cf)[0] for cf in temp_cfs]
                else:
                    temp_cfs = self.round_off_cfs(assign=False)
                    test_preds = [self.predict_fn(cf)[0] for cf in temp_cfs]

                if self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False

    def plot_grad_flow(self, named_parameters):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                print(n)
                print(p.grad)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        exit(1)
    
    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter,
                             max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
                             init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
                             posthoc_sparsity_algorithm, limit_steps_ls):
        """Finds counterfactuals by gradient-descent."""
        query_instance = self.model.transformer.transform(query_instance).to_numpy()[0]
        self.x1 = torch.tensor(query_instance).to(self.device)

        if self.decode_before_loss:
            self.x1_decoded = self.model.get_decoded(self.x1).to(self.device).detach()




        # find the predicted value of query_instance
        test_pred = self.predict_fn(self.x1.float())[0]



        if desired_class == "opposite":
            desired_class = 1.0 - np.round(test_pred)
        self.target_cf_class = torch.tensor(desired_class).to(self.device).float()

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

        # variables to backup best known CFs so far in the optimization process -
        # if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.min_dist_from_threshold = [100]*loop_find_CFs  # for backup CFs

        if self.plot_gradients:
            if self.decode_before_loss:
                distributions = [[] for dd in range(len(self.decoded_encoded_categorical_feature_indexes))] 
            else:
                distributions = [[] for dd in range(len(self.encoded_categorical_feature_indexes))] 

                regularization_losses = [[] for dd in range(len(self.encoded_categorical_feature_indexes))] 

                optimization_steps = [] 

                dx = 150

        for loop_ix in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                self.initialize_CFs(query_instance, False)
            else:
                self.initialize_CFs(query_instance, init_near_query_instance)

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()

                # get loss and backpropogate
                loss_value = self.compute_loss()


                if self.plot_gradients and (iterations) % dx == 0 and iterations>0:
                    for dd, v in enumerate(self.encoded_categorical_feature_indexes):
                        regularization_losses[dd].append(torch.pow((torch.sum(self.cfs[0][v[0]:v[-1]+1]) - 1.0), 2).cpu().item())

                    optimization_steps.append(iterations)



                if (iterations) == 0:

                    if self.plot_gradients:
                        if self.decode_before_loss:
                            for dd, dec_idx in enumerate(self.decoded_encoded_categorical_feature_indexes):
                                tensor_cpu = self.model.get_decoded(self.cfs[0])[dec_idx].detach().clone().cpu().numpy()
                                distributions[dd].append(tensor_cpu)
                        else:
                            for dd, dec_idx in enumerate(self.encoded_categorical_feature_indexes):
                                tensor_cpu = self.cfs[0][dec_idx].cpu().detach().numpy()

                                distributions[dd].append(tensor_cpu)

                        optimization_steps.append(0)
                        for dd, v in enumerate(self.encoded_categorical_feature_indexes):
                            regularization_losses[dd].append(0)
                


                self.loss.backward()





                if self.verbose and (iterations) % 50 == 0:
                    print('\tItteration: {}, Loss: {:.6f}, yloss: {:.6f}, proximity: {:.6f}, proximity latent: {:.6f}, plausibility: {:.6f}, diversity: {:.6f}, reguralization: {:.6f}'.format(iterations, loss_value, self.yloss, self.proximity_loss, self.proximity_loss_latent ,self.plausibility_loss_latent, self.diversity_loss, self.regularization_loss))


                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        if jx not in self.feat_to_vary_idxs:
                            self.cfs[ix].grad[jx] = 0.0

                if len(self.immutable_mask) > 0:
                    for ix in range(self.total_CFs):
                        for jx in range(len(self.minx[0])):
                            if jx not in self.immutable_mask:
                                self.cfs[ix].grad[jx] = 0.0
                    # print(self.cfs[0])
                    # print(self.cfs[0].grad)
                    # exit(1)
                # update the variables
                self.optimizer.step()

                if self.plot_gradients:
                    # pass



                    if (iterations) % dx == 0 and iterations > 0:

                        if self.plot_gradients:

                            if self.decode_before_loss:
                                for dd, dec_idx in enumerate(self.decoded_encoded_categorical_feature_indexes):
                                    tensor_cpu = self.model.get_decoded(self.cfs[0])[dec_idx].detach().clone().cpu().numpy()
                                    distributions[dd].append(tensor_cpu)
                            else:
                                for dd, dec_idx in enumerate(self.encoded_categorical_feature_indexes):
                                    tensor_cpu = self.cfs[0][dec_idx].cpu().detach().numpy()

                                    distributions[dd].append(tensor_cpu)


                # projection step
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        self.cfs[ix].data[jx] = torch.clamp(self.cfs[ix][jx], min=self.minx[0][jx], max=self.maxx[0][jx])

                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid

                if self.decode_before_loss:
                    temp_cfs_stored = self.decoded_round_off_cfs(assign=False)
                    test_preds_stored = [self.predict_fn_bb_only(cf) for cf in temp_cfs_stored]
                else:
                    temp_cfs_stored = self.round_off_cfs(assign=False)
                    test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]

                if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or
                   (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    avg_preds_dist = np.mean([abs(pred[0]-self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                        self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                        for ix in range(self.total_CFs):
                            self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(temp_cfs_stored[ix])
                            self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])




            if self.decode_before_loss:
                self.decoded_round_off_cfs(assign=True)
            else:
                self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                if self.decode_before_loss:
                    temp = self.cfs_input[j]
                    self.final_cfs.append(temp)
                else:
                    temp = self.cfs[j].detach().clone()#.detach().clone().cpu().numpy()
                    self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time

        if self.decode_before_loss:
            self.cfs_preds = [self.predict_fn_bb_only(cfs) for cfs in self.final_cfs]
        else:
            self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]


        # update final_cfs from backed up CFs if valid CFs are not found
        if ((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in self.cfs_preds))):
            for loop_ix in range(loop_find_CFs):
                if self.min_dist_from_threshold[loop_ix] != 100:
                    for ix in range(self.total_CFs):
                        self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                        self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])

        # convert to the format that is consistent with dice_tensorflow
        query_instance = np.array([query_instance], dtype=np.float32)
        for tix in range(max(loop_find_CFs, self.total_CFs)):
            self.final_cfs[tix] = np.array([self.final_cfs[tix].cpu().numpy()], dtype=np.float32)
            self.cfs_preds[tix] = np.array([self.cfs_preds[tix]], dtype=np.float32)

            # if self.final_cfs_sparse is not None:
            #     self.final_cfs_sparse[tix] = np.array([self.final_cfs_sparse[tix]], dtype=np.float32)
            #     self.cfs_preds_sparse[tix] = np.array([self.cfs_preds_sparse[tix]], dtype=np.float32)
            #
            if isinstance(self.best_backup_cfs[0], np.ndarray):  # checking if CFs are backed
                self.best_backup_cfs[tix] = np.array([self.best_backup_cfs[tix]], dtype=np.float32)
                self.best_backup_cfs_preds[tix] = np.array([self.best_backup_cfs_preds[tix]], dtype=np.float32)




        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])

        if self.decode_before_loss:
            final_cfs_df = self.dice_model_on_input.transformer.inverse_transform(
                    self.data_dice_input.get_decoded_data(cfs))
            
            # self.data_interface = self.data_dice_input
        else:
            final_cfs_df = self.model.transformer.inverse_transform(
                    self.data_interface.get_decoded_data(cfs))


        
        # rounding off to 3 decimal places
        cfs_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds]
        cfs_preds = [item for sublist in cfs_preds for item in sublist]
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)


        if not self.decode_before_loss:
            test_instance_df = self.model.transformer.inverse_transform(
                    self.data_interface.get_decoded_data(query_instance))
        else:

            # print(self.model.transformer.transform(self.test_instance_df))
            test_instance_df = self.test_instance_df
            # self.data_interface = self.data_dice_input
            
        test_instance_df[self.data_interface.outcome_name] = np.array(np.round(test_pred, 3))


        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        else:
            final_cfs_df_sparse = None

        m, s = divmod(self.elapsed, 60)
        if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))):
            self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            valid_ix = []  # indexes of valid CFs
            for cf_ix, pred in enumerate(self.cfs_preds):
                if ((self.target_cf_class == 0 and pred[0][0] < self.stopping_threshold) or
                   (self.target_cf_class == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1
                    valid_ix.append(cf_ix)

            if self.total_CFs_found == 0:
                print('No Counterfactuals found for the given configuation, ',
                      'perhaps try with different values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                      ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                      ' values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)

        if self.plot_gradients and len(valid_ix)>0:

            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from matplotlib.colors import LinearSegmentedColormap

            if self.decode_before_loss:
                forl = self.decoded_encoded_categorical_feature_indexes
            else:
                forl = self.encoded_categorical_feature_indexes

            labelsize = 20
            ticksize = 15
            space_between = 0.1
            for dd, dec_idx in enumerate(forl):
                categories = range(distributions[dd][0].shape[-1])  # Categories as a range

                # Colormap for the bars
                # cmap = plt.get_cmap('turbo')
                cmap = LinearSegmentedColormap.from_list("custom_greyscale", [(0.8, 0.8, 0.8), (0, 0, 0)], N=optimization_steps[-1])

                colors = cmap(np.linspace(0, 1, len(distributions[dd])))

                # Create a figure
                fig = plt.figure(figsize=(11, 6))  # Adjusted figure size for better control

                # Define GridSpec layout for the subplots
                gs = GridSpec(1, 3, width_ratios=[1, 0.05, 0.6], wspace=0.0)  # Set width_ratios to control space allocation

                # First subplot: Bar plot
                ax1 = fig.add_subplot(gs[0])
                bar_width = 0.15  # Width of the bars
                gap_width = 0.1   # Gap width between categories
                n_categories = len(categories)
                n_distributions = len(distributions[dd])

                for i, tensor in enumerate(distributions[dd]):
                    indices = np.arange(n_categories) * (bar_width * n_distributions + gap_width) + i * bar_width
                    ax1.bar(indices, tensor, width=bar_width, color=colors[i], align='center')

                # Set x-axis ticks and labels
                middle_bar_positions = np.arange(n_categories) * (bar_width * n_distributions + gap_width) + (bar_width * (n_distributions - 1)) / 2
                ax1.set_xticks(middle_bar_positions)

                x_tick_cat = [f'$c_{{{i}}}$' for i in range(n_categories)]

                ax1.set_xticklabels(x_tick_cat, fontsize=ticksize)


                ax1.set_yticks([0, 1])
                ax1.set_yticklabels(['0', '1'])

                # ax1.spines['left'].set_position('zero')
                # ax1.spines['right'].set_position('zero')

                xticks = ax1.get_xticks()


                # Add vertical lines between each category
                for jk, position in enumerate(xticks):
                    # if jk%2==0:
                    #     new_pos = position - n_distributions * bar_width / 2
                    # else:
                    new_pos = position - (n_distributions * bar_width + gap_width) / 2

                    ax1.axvline(x=new_pos, color='gray', linestyle='--', linewidth=1)

                ax1.axvline(x=position + (n_distributions * bar_width + gap_width) / 2, color='gray', linestyle='--', linewidth=1)
            
                # exit(1)
                ax1.set_xlabel('One-hot categories', fontsize=labelsize, labelpad=15)
                ax1.set_ylabel('One-hot distributions\nacross optimization steps', fontsize=labelsize)
                # ax1.set_title('Bar Plot of Tensors with Vertical Lines and Gaps')

                # Color bar next to the bar plot
                ax_cbar = fig.add_subplot(gs[1])
                norm = plt.Normalize(vmin=0, vmax=len(distributions[dd])-1)
                cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar, orientation='vertical')
                ax_cbar.yaxis.set_ticks_position('left')

                # Set color bar ticks based on optimization_steps
                cbar_ticks = np.linspace(0, len(optimization_steps) - 1, num=len(optimization_steps))
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f"{int(step)}" for step in optimization_steps])
                cbar.ax.yaxis.set_label_position('left')
                cbar.set_label('Optimization Steps', labelpad=10, fontsize=labelsize) 


                # Second subplot: Horizontal bar plot
                ax2 = fig.add_subplot(gs[2])
                n_losses = len(regularization_losses[dd])
                indices = np.arange(n_losses)  # Positions for the bars

                # Use colors for the bars, matching the color bar
                ax2.barh(indices, regularization_losses[dd], height=bar_width, color=colors)

                # Align the y-axis of the second plot with the color bar's y-axis
                cbar_ticks = ax_cbar.get_yticks()
                y_ticks = np.linspace(0, len(regularization_losses[dd]) - 1, len(cbar_ticks))
                ax2.set_yticks(y_ticks)
                ax2.set_ylim(ax_cbar.get_ylim())  # Set the y-axis limits of ax2 to match the color bar
                ax2.set_yticklabels([])

                ax2.set_xlabel('One-hot regularization\nloss', fontsize=labelsize, labelpad=15)

                ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

                ax2.spines['bottom'].set_position('zero')

                # Adjust the position of ax_cbar manually to control the gap specifically
                pos = ax_cbar.get_position()
                ax_cbar.set_position([pos.x0 + space_between, pos.y0, pos.width, pos.height])  # Increase x0 to adjust gap

                pos_ax2 = ax2.get_position()

                ax2.set_position([pos_ax2.x0 + space_between, pos_ax2.y0, pos_ax2.width, pos_ax2.height])  # Increase x0 to adjust gap


                # plt.show()
                # exit(1)
                # Save the combined figure
                if self.decode_before_loss:
                    plt.savefig(f'test/distributions/us/one_hot_distr_sample_{self.index}_feature_{dd}.png', format='png', dpi=300, bbox_inches='tight')
                else:
                    plt.savefig(f'test/distributions/dice/one_hot_distr_sample_{self.index}_feature_{dd}.png', format='png', dpi=300, bbox_inches='tight')

                plt.close(fig)  # Close the figure to save memory

        # returning only valid CFs
        return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse
