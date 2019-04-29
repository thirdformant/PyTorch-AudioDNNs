import os
import logging
import argparse
from typing import Collection, Optional, Tuple, Union
from config import *


__all__ = ['parse_arguments', 'init_logger', 'files_from_dir']


#==================
# Logging
#==================
def init_logger(logger:logging.Logger, output:str,
                verbose:bool=False):
    stream_handler = logging.StreamHandler()
    if verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s:%(message)s')
    stream_handler.setFormatter(ch_formatter)

    file_handler = logging.FileHandler(output)
    fh_formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
    file_handler.setFormatter(fh_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


#==================
# Directories
#==================
def make_dir(path:str)->str:
    """
    Creates the directory if it does not exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


OUTPUT_PATH = make_dir(OUTPUT_PATH)


def files_from_dir(audio_dir:str)->Tuple[Collection[str],
                                   Collection[Union[str, int]]]:
    files = []
    labels = []
    dir_names = [d for d in os.listdir(audio_dir)
                if os.path.isdir(os.path.join(audio_dir, d))]
    for label in dir_names:
        file_names = [os.path.join(audio_dir, label, filenames)
                     for filenames in os.listdir(os.path.join(audio_dir, label))
                     if filenames.lower().endswith(".wav")]
        for fname in file_names:
            files.append(fname)
            labels.append(label)
    return files, labels


#==================
# CLI parser
#==================
def parse_arguments()->dict:
    """
    Get arguments from command line
    """
    parser = argparse.ArgumentParser(description="Train either the Conv-3 or Conv-5 models from Huzaifah (2017) on a desired STFT feature set.")

    parser.add_argument('-input_dir', '--input_dir', dest='input_dir',
        type=str, default=INPUT_PATH, help='Path to directory of input audio files')
    parser.add_argument('-output_dir', '--output_dir', dest='output_dir',
        type=str, default=OUTPUT_PATH, help='Path to directory where trained model will be saved.')
    parser.add_argument('-ft', '--feature_type', dest='feature_type',
        type=str, default=FEATURE_TYPE, choices=['wideband_linear', 'narrowband_linear', 'wideband_mel', 'narrowband_mel'],
        help="""
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
