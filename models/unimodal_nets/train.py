import torch
import torch.nn as nn
import torch.optim as optim

def train_decoder(model, train_loader, device, epochs=5):
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.decoder.parameters(), lr=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    print("Entrenamiento inicial (solo decoder)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.float().to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def fine_tune_encoder(model, train_loader, device, epochs=5):
    for layer in model.encoder.children():
        for param in list(layer.parameters())[-10:]:
            param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    print("Fine-Tuning: Descongelando últimas capas del encoder...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.float().to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
        avg_loss = running_loss / len(train_loader)
        print(f"Fine-Tuning Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Modelo Fine-Tuned guardado exitosamente.")
