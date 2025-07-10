from ultralytics import SAM

def load_sam2(model_path="sam2.1_b.pt", device="cuda"):
    model = SAM(model_path)
    return model.to(device)
