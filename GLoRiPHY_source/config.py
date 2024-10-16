import argparse
import numpy as np

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--free_gpu_id',
                        type=int,
                        default=0,
                        help='The selected gpu.')
         
    parser.add_argument('--train_denoiseGenCore',
                        action='store_true',
                        default=False,
                        help='Choose whether to train the denoiser.')
    
    parser.add_argument('--train_denoiseGen',
                        action='store_true',
                        default=False,
                        help='Choose whether to train the denoiser.')
    
    parser.add_argument('--real_data',
                        action='store_true',
                        default=False,
                        help='Choose whether to run the script in test mode.')
    
    parser.add_argument('--sim_data',
                        action='store_true',
                        default=False,
                        help='Choose whether to run the script in test mode.')

    # LoRa Parameters
    parser.add_argument('--sf',
                        type=int,
                        default=8,
                        help='The spreading factor.')
    parser.add_argument('--fs_bw_ratio',
                        type=int,
                        default=8,
                        help='Ratio sampling rate vs BW.')
    parser.add_argument("--snr_list", 
                        nargs='+', 
                        default=list(range(-10, 10)), 
                        type=int)  # for train: -25:0, test: -40, 16
    
    # Input Transform
    parser.add_argument('--dechirp',
                        type = bool,
                        default = False,
                        help = 'Choose whether to dechirp or not')
    parser.add_argument('--x_image_channel', 
                        type=int, 
                        default=2)
    parser.add_argument('--normalization',
                        action='store_true',
                        default=True,
                        help='Choose whether to include the cycle consistency term in the loss.')

    # Conformer hyper-parameters
    parser.add_argument('--transformer_encoder_dim',
                        type=int,
                        default=256)
    parser.add_argument('--num_attention_heads',
                        type=int,
                        default=16)
    parser.add_argument('--transformer_layers',
                        type=int,
                        default=1)
    parser.add_argument('--attention_dropout',
                        type=float,
                        default=0.2)
    parser.add_argument('--inp_dropout',
                        type=float,
                        default=0.15)
    parser.add_argument('--ff_dropout',
                        type=float,
                        default=0.15)
    parser.add_argument('--conv_dropout',
                        type=float,
                        default=0.15)
    
    # Training hyper-parameters
    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--init_zero_weights',
                        action='store_true',
                        default=False,
                        help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')
    parser.add_argument('--sched_factor',
                        type=float,
                        default=0.7,
                        help='The learning rate scheduler factor.')
    parser.add_argument('--sched_patience',
                        type=int,
                        default=7,
                        help='The learning rate scheduler patience.')
    parser.add_argument('--mse_scaling',
                        type=float,
                        default=7700,
                        help='The scaling factor for the MSE loss.')

    # Training settings
    parser.add_argument('--test_mode',
                        action='store_true',
                        default=False,
                        help='Choose whether to run the script in test mode.')
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=200,
                        help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--load_epoch',
                        type=int,
                        default=-1,
                        help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--load', 
                        type=str, 
                        default=None)
    parser.add_argument('--load_symbol_conformer',
                        type=str,
                        default = None,
                        help='The path to load the pretrained conformer model.')

    parser.add_argument('--checkpoint_every', 
                        type=int, 
                        default=5)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='The number of images in a batch.')
    parser.add_argument('--train_data_size',
                        type=int,
                        default=1000000,
                        help='The number of training images.')
    parser.add_argument('--test_data_size',
                        type=int,
                        default=5000,
                        help='The number of test images.')
    parser.add_argument('--phase1',
                        type=int,
                        default=200,
                        help='The number of epochs Phase 1 lasts for')
    parser.add_argument('--phase2',
                        type=int,
                        default=400,
                        help='The number of epochs Phase 2 lasts for')
    parser.add_argument('--phase3',
                        type=int,
                        default=600,
                        help='The number of epochs Phase 3 lasts for')
    parser.add_argument('--early_stopping_patience',
                        type=int,
                        default=50,
                        help='The patience for early stopping')      
    parser.add_argument('--num_workers',
                        type=int,
                        default=13,
                        help='The number of threads to use for the DataLoader.')

    # Data sources
    parser.add_argument('--root_path',
                        type=str,
                        default = None,
                        help='Choose the root path to the code.')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default= None,)
    parser.add_argument('--data_dir',
                        type=str,
                        default= None,
                        help='Choose the path to the data.')
    
    #Data Settings
    parser.add_argument('--num_packets',
                        type=int,
                        default=50,
                        help='The number of packets to be used for training.')
    parser.add_argument('--num_perturbations',
                        type=int,
                        default=1000,
                        help='The number of perturbations.')
    
    # Testing Choice
    parser.add_argument('--test_choice',
                        type=str,
                        default='LoraPHY',
                        help='Choose the type of test to run.')
    parser.add_argument('--test_nodes',
                        nargs='+',
                        type=int,
                        default=[0])
    
    
    return parser