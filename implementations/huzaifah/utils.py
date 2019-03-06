import os
import argparse
from config import *


__all__ = ['parse_arguments']


def make_dir(path):
    """
    Creates the directory if it does not exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


OUTPUT_PATH = make_dir(OUTPUT_PATH)


def parse_arguments():
    """
    Get arguments from command line
    """
    parser = argparse.ArgumentParser(description="Train either the Conv-3 or Conv-5 models from Huzaifah (2017) on a desired STFT feature set.")

    parser.add_argument('-input_dir', '--input_dir', dest='input_dir',
        type=str, default=INPUT_PATH, help='Path to directory of input audio files')
    parser.add_argument('-output_dir', '--output_dir', dest='output_dir',
        type=str, default=OUTPUT_PATH, help='Path to directory where trained model will be saved.')
    parser.add_argument('-ft', '--feature_type', dest='feature_type',
        type=str, default=FEATURE_TYPE, help="""
        The type of audio feature to be used for training, where
            'wideband_linear': Wideband linear-scaled STFT spectrogram;
            'narrowband_linear': Narrowband linear-scaled STFT spectrogram;
            'wideband_mel': Wideband mel-scaled STFT spectrogram;
            'narrowband_mel': Narrowband mel-scaled STFT spectrogram.
        """)
    parser.add_argument('-arch', '--model_architecture', dest='model_arch',
        type=str, default=MODEL_ARCH, help="The model architecture to use, either 'conv-3' or 'conv-5'.")
    parser.add_argument('-vr', '--validation_ratio', dest='valid_ratio',
        type=float, default=VALID_RATIO, help='Ratio of input audio files to be used for validation during training.')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int,
        default=BATCH_SIZE, help='Batch size used for training.')
    parser.add_argument('-ne', '--number_epochs', dest='num_epochs', type=int,
        default=NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate',
        type=float, default=LEARNING_RATE, help='Learning rate for ADAM optimizer.')
    parser.add_argument('-rf', '--regularization_factor', dest='reg_factor',
        type=float, default=REGULARIZATION_FACTOR, help='Regularization factor for gradient penalty.')
    args = parser.parse_args()
    return vars(args)
