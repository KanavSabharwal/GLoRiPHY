import argparse

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--free_gpu_id',
                        type=int,
                        default=1,
                        help='The selected gpu.')

    parser.add_argument('--x_image_channel', type=int, default=2)
    parser.add_argument('--y_image_channel', type=int, default=2)
    parser.add_argument('--conv_kernel_size', type=int, default=3)
    parser.add_argument('--conv_padding_size', type=int, default=1)
    parser.add_argument('--lstm_dim', type=int, default=400)  # For mask_CNN model
    parser.add_argument('--fc1_dim', type=int, default=600)  # For mask_CNN model

    parser.add_argument('--sf',
                        type=int,
                        default=8,
                        help='The spreading factor.')
    parser.add_argument('--fs_bw_ratio',
                        type=int,
                        default=8,
                        help='Ratio sampling rate vs BW.')
    parser.add_argument('--bw',
                        type=int,
                        default=125000,
                        help='The bandwidth.')
    parser.add_argument('--fs',
                        type=int,
                        default=1000000,
                        help='The sampling rate.')
    parser.add_argument('--normalization',
                        action='store_true',
                        default=True,
                        help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='The number of images in a batch.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=13,
                        help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help='The learning rate (default 0.0003)')
    parser.add_argument('--scaling_for_imaging_loss',
                        type=int,
                        default=128,
                        help='The scaling factor for the imaging loss')
    parser.add_argument('--scaling_for_classification_loss',
                        type=int,
                        default=1,
                        help='The scaling factor for the classification loss')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--root_path',
                        type=str,
                        default = None,
                        help='Choose the root path to the code.')
    parser.add_argument('--data_dir',
                        type=str,
                        default = None,
                        help='Choose the root path to rf signals.')

    parser.add_argument("--snr_list", nargs='+', default=list(range(-25, 16)), type=int)  # for train: -25:0, test: -40, 16
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=1)
    
    parser.add_argument('--test_mode',
                        action='store_true',
                        default=False,
                        help='Choose whether to run the script in test mode.')    
    parser.add_argument('--sim_data',
                        action='store_true',
                        default=False)
    
    parser.add_argument('--real_data',
                        action='store_true',
                        default=False)
    
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
    parser.add_argument('--early_stopping_patience',
                        type=int,
                        default=20,
                        help='The patience for early stopping') 

    parser.add_argument('--num_packets',
                        type=int,
                        default=120,
                        help='The number of packets to be used for training.')
    parser.add_argument('--num_perturbations',
                        type=int,
                        default=1000,
                        help='The number of perturbations.')   
    parser.add_argument('--train_data_size',
                        type=int,
                        default=1000000,
                        help='The number of training images.')
    parser.add_argument('--test_data_size',
                        type=int,
                        default=100000,
                        help='The number of test images.')
    parser.add_argument('--test_nodes',
                        nargs='+',
                        type=int,
                        default=[0])
    return parser
