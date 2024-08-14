import torch

# Paths
DATASET_PATH = './dataset'
OUTPUT_PATH = './output'
MODELS_PATH = './output_models'

# Hyperparameters
IMAGE_SIZE = 64
BATCH_SIZE = 128
LATENT_SIZE = 128
NGF = 64
NDF = 64
NC = 3
EPOCHS = 5
LEARNING_RATE = 3e-4
BETAS = (0.5, 0.999)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
