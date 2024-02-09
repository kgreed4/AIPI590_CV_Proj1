from data_augmentation import augment_data, balance_dataset
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, base_dir='./', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.base_dir, self.data_frame.iloc[idx, 0])
        
        image = Image.open(img_name).convert('RGB')
        
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(train_csv, val_csv, test_csv, base_dir='./', augment=False, balance_classes=False):
    # Apply any transformations here
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # augment training data
    if augment:
        csv_output = 'data-csvs/augmented_dataset.csv'
        augment_data(train_csv, csv_output, output_folder='augmented_dataset')
        train_csv = csv_output

    # balance the classes
    if balance_classes:
        csv_output = 'data-csvs/balanced_augmented_dataset.csv'
        balance_dataset(train_csv, csv_output, output_folder='balanced_augmented_dataset')
        train_csv = csv_output

    train_dataset = CustomDataset(csv_file=train_csv, base_dir=base_dir, transform=transform)
    val_dataset = CustomDataset(csv_file=val_csv, base_dir=base_dir, transform=transform)
    test_dataset = CustomDataset(csv_file=test_csv, base_dir=base_dir, transform=transform)

    return train_dataset, val_dataset, test_dataset

def setup_dataloaders(train_csv='data-csvs/train_images_labeled.csv', val_csv='data-csvs/valid_images_labeled.csv', test_csv='data-csvs/test_images_labeled.csv', batch_size=32):
    train_dataset, val_dataset, test_dataset = load_data(train_csv, val_csv, test_csv, augment=False, balance_classes=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Test the dataloaders
if __name__ == "__main__":
    train_loader, val_loader, test_loader = setup_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i} - Images: {len(images)}, Labels: {len(labels)}")
        if i == 2:
            break
