# inference.py
import torch, numpy as np
from models.ae_3d.autoencoder_model import *
from models.ae_3d.config import cfg

def load_best_model(path, device=cfg.device):
    model = AutoEncoder(cfg.in_modalities, cfg.latent_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval().to(device)
    return model

def reconstruct_volume(model, volume: np.ndarray):
    with torch.no_grad():
        x = torch.from_numpy(volume).unsqueeze(0).to(cfg.device)
        rec = model(x).squeeze(0).cpu().numpy()
    return rec
