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
