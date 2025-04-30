import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
# BATCH_SIZE          = 4096
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
# NUM_EPOCHS          = 10
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.001, 'momentum': 0.9}
# OPTIMIZER_PARAMS    = {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8}
OPTIMIZER_PARAMS       = {'type': 'AdamW', 'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01, 'eps': 1e-8}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
# SCHEDULER_PARAMS    = {'type': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.1, 'patience': 5, 'min_lr': 1e-6}
# SCHEDULER_PARAMS    = {'type': 'CyclicLR', 'base_lr': 1e-4, 'max_lr': 0.01, 'step_size_up': 2000, 'mode': 'triangular2'}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
# NUM_WORKERS         = 8
NUM_WORKERS         = 0

# Augmentation
'''
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]
'''
# Augmentation2
'''
IMAGE_ROTATION      = 30
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 16
IMAGE_PAD_CROPS     = 8
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]
'''
# Augmentation3
# Augmentation
IMAGE_ROTATION      = 40
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 32
IMAGE_PAD_CROPS     = 3
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'resnet18'
# MODEL_NAME          = 'resnet101'
# MODEL_NAME          = 'efficientnet_b0'
# MODEL_NAME          = 'alexnet'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
