# config.py
from pathlib import Path
import torch, numpy as np, random

class AEConfig:
    seed              = 55
    in_modalities     = 4
    latent_dim        = 512
    batch_size        = 1
    num_epochs        = 50
    lr                = 5e-4
    accum_steps       = 4
    path_to_csv         = Path("./data/train_data.csv")
    data_root         = Path(
        "./data/brats20-dataset-training-validation/"
        "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    )
    ae_pretrained_model_path   = Path("./models/brats20logs/brats2020logs/ae/autoencoder_best_model.pth")   # opcional
    save_dir          = Path("./models/brats20logs/brats20logs/ae/")
    device            = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

cfg = AEConfig()
seed_everything(cfg.seed)
