import numpy as np
import torch

class GlobalConfig:
    root_dir = './data/brats20-dataset-training-validation'
    train_root_dir = './data/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = './data/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = './data/train_data.csv'
    pretrained_model_path = None
    train_logs_path = None
    tab_data = None
    seed = 55

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
