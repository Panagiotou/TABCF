# from baselines.great.main import main as train_great
# from baselines.goggle.main import main as train_goggle
# from baselines.codi.main import main as train_codi
# from baselines.stasy.main import main as train_stasy
# from baselines.tabddpm.main_train import main as train_tabddpm
# from baselines.smote.main import main as train_smote

# from baselines.great.sample import main as sample_great
# from baselines.goggle.sample import main as sample_goggle
# from baselines.codi.sample import main as sample_codi
# from baselines.stasy.sample import main as sample_stasy
# from baselines.tabddpm.main_sample import main as sample_tabddpm

from tabcf.vae.main import main as train_vae
from tabcf.vae.train_black_box import main as train_black_box
# from tabcf.main import main as train_tabcf
from tabcf.sample import main as sample_tabcf
from evaluation_framework.evaluate import main as evaluate
from baselines.dice.sample import main as sample_dice
from baselines.revise.sample import main as sample_revise
from baselines.cchvae.sample import main as sample_cchvae
from baselines.wachter.sample import main as sample_wachter





import argparse
import importlib





def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def execute_function(method, mode):
    if method == 'vae':
        mode = 'train'

    if mode == "evaluate":
        main_fn = eval("evaluate")
    else:
        if method == "dice":
            mode = "sample"
        main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabcf', help='Method: tabcf or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Verbose')


    # configs for traing tabcf's VAE
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')
    parser.add_argument('--gumbel_softmax', type=str2bool, default=True, help='Gumbel softmax output (differentiable)?')
    parser.add_argument('--tau', type=float, default=1.0, help='Tau param')
    parser.add_argument('--sigmoid', type=str2bool, default=True, help='Sigmoid the continuous during decoding')
    parser.add_argument('--num_reconstr_loss', type=str, default="L2", help='Reconstructions loss for continuous part')
    parser.add_argument('--reparam', type=str2bool, default=True, help='Use mean + std latents')

    parser.add_argument('--kl_weight', type=float, default=-1, help='KL Weight increasing beta')

    parser.add_argument('--hidden_dims', type=int, default=16, help='Hidden dims of black-box model')


    parser.add_argument('--plot_gradients', type=str2bool, default=False, help='For plots optimization')



    # configs for sampling
    parser.add_argument('--save_path', type=str, default="counterfactual_results", help='Path to save synthetic data.')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of test samples to find CFs.')

    parser.add_argument('--dice_optimization', type=str2bool, default=True, help='Use dice latent optim')
    parser.add_argument('--num_encoding', type=str, default="min_max_torch", help='Encoding Num data (for VAE)')
    parser.add_argument('--decode_before_loss', type=str2bool, default=True, help='Proximity loss in the output space')

    parser.add_argument('--proximity_weight_input', type=float, default=1.0, help='Proximity weight (loss calculated in input space)')
    parser.add_argument('--proximity_weight_latent', type=float, default=1.0, help='Proximity weight (loss calculated in latent space)')
    parser.add_argument('--proximity_latent_loss', type=str, default="L2", help='Proximity loss (loss calculated in latent space)')
    parser.add_argument('--proximity_input_loss', type=str, default=None, help='Proximity loss (loss calculated in input space)')
    parser.add_argument('--plausibility_weight_latent', type=float, default=-1.0, help='Plausibility loss weight (loss calculated in latent space)')
    parser.add_argument('--plausibility_latent_loss', type=str, default="GMM", help='Plausibility loss (loss calculated in latent space)')

    parser.add_argument('--immutable', type=str2bool, default=False, help='Consider immutable features')


    parser.add_argument('--total_CFs', type=int, default=1, help='How many CFs to find for each instance')

    parser.add_argument('--min_iter', type=int, default=500, help='Min GD iterations')
    parser.add_argument('--max_iter', type=int, default=5000, help='Max GD iterations')


    # parser.add_argument('--repeats', type=int, default=None, help='Repeat experiments')



    #GMM --------------------
    parser.add_argument("--samples", type=int, default=1000, help="The number of total samples in dataset")
    parser.add_argument("--components", type=int, default=5, help="The number of gaussian components in mixture model")
    parser.add_argument("--dims", type=int, default=2, help="The number of data dimensions")
    parser.add_argument("--iterations", type=int, default=20_000, help="The number optimization steps")
    parser.add_argument("--family", type=str, default="full", help="Model family, see `Mixture Types`")
    parser.add_argument("--log_freq", type=int, default=5_000, help="Steps per log event")
    parser.add_argument("--radius", type=float, default=8.0, help="L1 bound of data samples")
    parser.add_argument("--mixture_lr", type=float, default=3e-5, help="Learning rate of mixture parameter (pi)")
    parser.add_argument("--component_lr", type=float, default=1e-2, help="Learning rate of component parameters (mus, sigmas)")
    parser.add_argument("--visualize", type=bool, default=False, help="True for visualization at each log event and end")
    parser.add_argument("--seed", type=int, default=42, help="seed for numpy and torch")




    # parser.add_argument('--ohe_min_max', type=str2bool, default=True, help='Use one hot opt for training black box clf')
    # parser.add_argument('--latent_clf', type=str2bool, default=False, help='Use latent clf (competitor)')
    # parser.add_argument('--steps', type=int, default=200, help='NFEs.')
    # parser.add_argument('--validity', type=str2bool, default=True, help='Validity loss')
    # parser.add_argument('--proximity', type=str2bool, default=True, help='Proximity loss')
    # parser.add_argument('--sparsity', type=str2bool, default=True, help='Sparsity loss')
    # parser.add_argument('--diffusion', type=str2bool, default=False, help='Perform diffusion steps')

    ''' configs for dice '''

    parser.add_argument('--dice_method', type=str, default="gradient", help='Which dice method to use')
    parser.add_argument('--dice_post_hoc_sparsity', type=str2bool, default=False, help='Use dice post processing on resulting CFs (sparsity)')
    

    ''' configs for valuation '''

    parser.add_argument('--pyod', type=str2bool, default=True, help='Use pyod for outlier detection')
    parser.add_argument('--get_stats', type=str2bool, default=True, help='Print stats also')
    parser.add_argument('--get_changed', type=str2bool, default=False, help='get features changed')

    

    args = parser.parse_args()

    return args

    # ''' configs for CTGAN '''

    # parser.add_argument('-e', '--epochs', default=1000, type=int,
    #                     help='Number of training epochs')
    # parser.add_argument('--no-header', dest='header', action='store_false',
    #                     help='The CSV file has no header. Discrete columns will be indices.')

    # parser.add_argument('-m', '--metadata', help='Path to the metadata')
    # parser.add_argument('-d', '--discrete',
    #                     help='Comma separated list of discrete columns without whitespaces.')
    # parser.add_argument('-n', '--num-samples', type=int,
    #                     help='Number of rows to sample. Defaults to the training data size')

    # parser.add_argument('--generator_lr', type=float, default=2e-4,
    #                     help='Learning rate for the generator.')
    # parser.add_argument('--discriminator_lr', type=float, default=2e-4,
    #                     help='Learning rate for the discriminator.')

    # parser.add_argument('--generator_decay', type=float, default=1e-6,
    #                     help='Weight decay for the generator.')
    # parser.add_argument('--discriminator_decay', type=float, default=0,
    #                     help='Weight decay for the discriminator.')

    # parser.add_argument('--embedding_dim', type=int, default=1024,
    #                     help='Dimension of input z to the generator.')
    # parser.add_argument('--generator_dim', type=str, default='1024,2048,2048,1024',
    #                     help='Dimension of each generator layer. '
    #                     'Comma separated integers with no whitespaces.')
    # parser.add_argument('--discriminator_dim', type=str, default='1024,2048,2048,1024',
    #                     help='Dimension of each discriminator layer. '
    #                     'Comma separated integers with no whitespaces.')

    # parser.add_argument('--batch_size', type=int, default=500,
    #                     help='Batch size. Must be an even number.')
    # parser.add_argument('--save', default=None, type=str,
    #                     help='A filename to save the trained synthesizer.')
    # parser.add_argument('--load', default=None, type=str,
    #                     help='A filename to load a trained synthesizer.')

    # parser.add_argument('--sample_condition_column', default=None, type=str,
    #                     help='Select a discrete column name.')
    # parser.add_argument('--sample_condition_column_value', default=None, type=str,
    #                     help='Specify the value of the selected discrete column.')
    # parser.add_argument('--train_black_box', default=True, type=bool,
    #                     help='Train a black box clf on the embeddings')
    # ''' configs for GReaT '''

    # parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')

    # ''' configs for CoDi '''

    # # General Options
    # parser.add_argument('--logdir', type=str, default='./codi_exp', help='log directory')
    # parser.add_argument('--train', action='store_true', help='train from scratch')
    # parser.add_argument('--eval', action='store_true', help='load ckpt.pt and evaluate')

    # # Network Architecture
    # parser.add_argument('--encoder_dim', nargs='+', type=int, help='encoder_dim')
    # parser.add_argument('--encoder_dim_con', type=str, default="512,1024,1024,512", help='encoder_dim_con')
    # parser.add_argument('--encoder_dim_dis', type=str, default="512,1024,1024,512", help='encoder_dim_dis')
    # parser.add_argument('--nf', type=int, help='nf')
    # parser.add_argument('--nf_con', type=int, default=16, help='nf_con')
    # parser.add_argument('--nf_dis', type=int, default=64, help='nf_dis')
    # parser.add_argument('--input_size', type=int, help='input_size')
    # parser.add_argument('--cond_size', type=int, help='cond_size')
    # parser.add_argument('--output_size', type=int, help='output_size')
    # parser.add_argument('--activation', type=str, default='relu', help='activation')

    # # Training
    # parser.add_argument('--training_batch_size', type=int, default=4096, help='batch size')
    # parser.add_argument('--eval_batch_size', type=int, default=2100, help='batch size')
    # parser.add_argument('--T', type=int, default=50, help='total diffusion steps')
    # parser.add_argument('--beta_1', type=float, default=0.00001, help='start beta value')
    # parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    # parser.add_argument('--lr_con', type=float, default=2e-03, help='target learning rate')
    # parser.add_argument('--lr_dis', type=float, default=2e-03, help='target learning rate')
    # parser.add_argument('--total_epochs_both', type=int, default=20000, help='total training steps')
    # parser.add_argument('--grad_clip', type=float, default=1., help="gradient norm clipping")
    # parser.add_argument('--parallel', action='store_true', help='multi gpu training')

    # # Sampling
    # parser.add_argument('--sample_step', type=int, default=2000, help='frequency of sampling')

    # # Continuous diffusion model
    # parser.add_argument('--mean_type', type=str, default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
    # parser.add_argument('--var_type', type=str, default='fixedsmall', choices=['fixedlarge', 'fixedsmall'], help='variance type')

    # # Contrastive Learning
    # parser.add_argument('--ns_method', type=int, default=0, help='negative condition method')
    # parser.add_argument('--lambda_con', type=float, default=0.2, help='lambda_con')
    # parser.add_argument('--lambda_dis', type=float, default=0.2, help='lambda_dis')
    # ################    


    # # configs for TabDDPM
    # parser.add_argument('--ddim', action = 'store_true', default=False, help='Whether use DDIM sampler')

    # # configs for SMOTE
    # parser.add_argument('--cat_encoding', type=str, default='one-hot', help='Encoding method for categorical features')





    
