import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
# BATCH_SIZE          = 256
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 100
# NUM_EPOCHS          = 40
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.001, 'momentum': 0.9}
# OPTIMIZER_PARAMS    = {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8}
# OPTIMIZER_PARAMS       = {'type': 'AdamW', 'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01, 'eps': 1e-8}

OPTIMIZER_PARAMS       = {'type': 'AdamW', 'lr': 0.0005, 'betas': (0.9, 0.999), 'weight_decay': 0.05, 'eps': 1e-8}

# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
SCHEDULER_PARAMS      = {'type': 'CosineAnnealingLR','T_max': 50,'eta_min': 1e-6}
# SCHEDULER_PARAMS    = {'type': 'OneCycleLR', 'max_lr': 1e-3, 'epochs': 50, 'steps_per_epoch': 100,}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
# NUM_WORKERS         = 8
NUM_WORKERS         = 0

# for SOTA Method
# IMAGE_ROTATION = 30                        
# IMAGE_FLIP_PROB = 0.5                      
# IMAGE_NUM_CROPS = 64                       
# IMAGE_PAD_CROPS = 8                        
# IMAGE_MEAN = [0.4802, 0.4481, 0.3975]      
# IMAGE_STD = [0.2302, 0.2265, 0.2262]       
# RANDOM_ERASE_PROB = 0.25                                         
# COLOR_JITTER = (0.4, 0.4, 0.4, 0.1) 

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]
# Augmentation2
# IMAGE_ROTATION      = 30
# IMAGE_FLIP_PROB     = 0.5
# IMAGE_NUM_CROPS     = 16
# IMAGE_PAD_CROPS     = 8
# IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
# IMAGE_STD           = [0.2302, 0.2265, 0.2262]
# Augmentation3
# IMAGE_ROTATION      = 40
# IMAGE_FLIP_PROB     = 0.5
# IMAGE_NUM_CROPS     = 32
# IMAGE_PAD_CROPS     = 3
# IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
# IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
# MODEL_NAME          = 'resnet18'
# MODEL_NAME          = 'efficientnet_b0'
# MODEL_NAME          = 'alexnet'
# MODEL_NAME          = 'MyNetwork'
# MODEL_NAME          = 'efficientnet_b2'
# MODEL_NAME          = 'efficientnet_b3'
# MODEL_NAME          = 'efficientnet_b4'
# MODEL_NAME          = 'efficientnet_b5'
# MODEL_NAME          = 'efficientnet_b6' # batch_size 256
# MODEL_NAME          = 'efficientnet_b7' # batch_size 256
# MODEL_NAME          = 'resnet34'
# MODEL_NAME          = 'resnet50'
# MODEL_NAME          = 'resnet101'
# MODEL_NAME          = 'resnet152'
MODEL_NAME          = 'SOTA_MyNetwork'

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
