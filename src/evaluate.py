import torch
import torch.nn as nn
from src.model import TrafficSignCNN
from src.preprocessing import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model_path='models/best_traffic_sign_model.pth', data_dir='data', batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Loading model from {model_path}...")
    model = TrafficSignCNN(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get only test loader
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('logs/confusion_matrix.png')
    print("Confusion matrix saved to logs/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
