import torchvision
import os
import ssl

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def download_gtsrb(data_dir='data'):
    print(f"Downloading GTSRB dataset to {data_dir}...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download training set
    torchvision.datasets.GTSRB(root=data_dir, split='train', download=True)
    # Download test set
    torchvision.datasets.GTSRB(root=data_dir, split='test', download=True)
    print("Download complete.")

if __name__ == "__main__":
    download_gtsrb()
