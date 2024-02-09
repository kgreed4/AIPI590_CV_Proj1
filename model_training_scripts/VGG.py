import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from dataloading import setup_dataloaders


def initialize_model(num_classes=11):
    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace the classifier of the VGG model
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print('Training complete')
    return model

if __name__ == "__main__":
    num_classes = 11
    model = initialize_model(num_classes)
    print(model)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to {device}")

    # Define the loss function and optimizer
    # set up data loaders

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    # Train the model
    model_ft = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

