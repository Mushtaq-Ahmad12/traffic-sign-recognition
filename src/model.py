import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        self.dropout1 = nn.Dropout(0.25)
        
        # Convolutional Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8
        self.dropout2 = nn.Dropout(0.25)
        
        # Convolutional Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2) # 8x8 -> 4x4
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # FC layers
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test model with a dummy input
    model = TrafficSignCNN()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 43), "Output shape should be (1, 43)"
    print("Model initialized successfully and passed shape test.")
