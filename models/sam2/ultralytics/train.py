import torch
import torch.nn as nn

def freeze_image_encoder(model):
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False

def unfreeze_mask_decoder(model):
    for param in model.model.sam_mask_decoder.parameters():
        param.requires_grad = True

def get_optimizer(model, lr=1e-4):
    return torch.optim.Adam(model.model.sam_mask_decoder.parameters(), lr=lr)

def train_sam2(model, dataloader, optimizer, epochs=10, device="cuda"):
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        for image, masks_list in dataloader:
            image = image.to(device)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for mask in masks_list:
                mask = mask.to(device)
                with torch.autocast(device, dtype=torch.float32):
                    results = model(image)
                    pred_mask = results[0].masks.data
                if pred_mask.shape == mask.shape:
                    loss = loss_fn(pred_mask, mask)
                    total_loss += loss.detach()
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {total_loss.item()}")
