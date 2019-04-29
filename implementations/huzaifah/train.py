import os
import logging
import datetime
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torchaudio.transforms
import torchaudio
from utils import *


#==================
# Setup
#==================
DATE_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


#==================
# Parameters
#==================
args = parse_arguments()
# Data
input_dir = args['input_dir']
output_dir = args['output_dir']
# Training
feature_type = args['feature_type']
model_arch = args['model_arch']
batch_size = args['batch_size']
num_epochs = args['num_epochs']
learning_rate = args['learning_rate']
reg_factor = args['reg_factor']


#==================
# Logging
#==================
LOGGER = logging.getLogger('huzaifah')
LOGGER.setLevel(logging.DEBUG)
LOGGER_OUTPUT = os.path.join(output_dir, DATE_TIME + '.log')
init_logger(LOGGER, LOGGER_OUTPUT)

LOGGER.info('Initialised logger')


#==================
# Transformations
#==================
if feature_type == 'wideband_linear':
    LOGGER.info('Using wideband linear-scaled STFT spectrograms')
    audio_transforms = transforms.Compose([
        torchaudio.transforms.Spectrogram(n_fft=2048),
        SpectrogramToDB()
    ])
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((154, 12), 4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
elif feature_type == 'narrowband_linear':
    LOGGER.info('Using narrowband linear-scaled STFT spectrograms')
    audio_transforms = transforms.Compose([
        torchaudio.transforms.Spectrogram(n_fft=512),
        SpectrogramToDB()
    ])
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((37, 50), 4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
elif feature_type == 'wideband_mel':
    LOGGER.info('Using wideband mel-scaled STFT spectrograms')
    wb_mel_transform = transforms.Compose([
        torchaudio.transforms.MelSpectrogram(n_fft=2048, n_mels=512)
    ])
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((154, 12), 4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
     ])
elif feature_type == 'narrowband_mel':
    LOGGER.info('Using narrowband mel-scaled STFT spectrograms')
    wb_mel_transform = transforms.Compose([
        torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=128)
    ])
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((37, 50), 4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
     ])
