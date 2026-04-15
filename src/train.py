import torch
import torch.nn as nn
import torch.optim as optim
from src.model import TrafficSignCNN
from src.preprocessing import get_dataloaders
import os
from tqdm import tqdm

def train_model(epochs=30, lr=0.001, batch_size=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
    device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"========================================")
    print(f"TRAINING ON DEVICE: {device_name}")
    print(f"========================================")
    
    # 1. Get dataloaders (using 30% subset as requested)
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, subset_fraction=0.3)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")
    
    # 2. Initialize model, loss, and optimizer
    model = TrafficSignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=train_loss/len(train_loader), acc=100.*correct/total)
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/best_traffic_sign_model.pth')
            print("Model saved!")

if __name__ == "__main__":
    train_model()
