import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_transforms(img_size=32, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir='data', batch_size=64, val_split=0.2, subset_fraction=None):
    # Training set
    full_train_dataset = datasets.GTSRB(
        root=data_dir, 
        split='train', 
        transform=get_transforms(train=True)
    )
    
    # Optional subsetting for speed
    if subset_fraction is not None and subset_fraction < 1.0:
        subset_size = int(len(full_train_dataset) * subset_fraction)
        remaining_size = len(full_train_dataset) - subset_size
        full_train_dataset, _ = random_split(
            full_train_dataset, [subset_size, remaining_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Validation split
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Overwrite val_dataset transform to not have augmentation
    # Note: custom dataset wrapper might be needed if transforms are bound to dataset object
    # In torchvision GTSRB, they are. So we'll keep it simple for now or use a Subset Strategy.
    
    # Test set
    test_dataset = datasets.GTSRB(
        root=data_dir, 
        split='test', 
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # This will fail if data isn't downloaded yet
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Data not ready yet: {e}")
