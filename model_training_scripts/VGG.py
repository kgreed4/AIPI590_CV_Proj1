import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from dataloading import setup_dataloaders
import copy
import matplotlib.pyplot as plt
import os


def initialize_model(num_classes=11):
    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False
    
    print(model.classifier)
    # Replace the classifier of the VGG model
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=15, patience=3, filename='best_VGG.pth'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0  # For early stopping
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # Save the loss and accuracy for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0  # Reset counter
                # Save the model
                print(f"Saving model to {filename}")
                torch.save(model.state_dict(), os.path.join('model_results', filename))
            elif phase == 'val':
                epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping")
            break

    print('Training complete. Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

if __name__ == "__main__":
    num_classes = 11
    model = initialize_model(num_classes)
    print(model)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to {device}")

    # set up data loaders
    train_loader, val_loader, test_loader = setup_dataloaders(balance_classes=True, augment=True)
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model_ft, train_loss, val_loss, train_acc, val_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=15, patience=4, filename='VGG_balanced_augmented.pth')

    # Save the training and validation losses to a txt file
    with open(os.path.join('model_results','VGG_training.txt'), 'w') as f:
        for i in range(len(train_loss)):
            f.write(f"Epoch {i+1} - Train Loss: {train_loss[i]} - Val Loss: {val_loss[i]} - Train Acc: {train_acc[i]} - Val Acc: {val_acc[i]}\n")

    # Plot the training and validation loss
    plt.figure(0)
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss for VGG16 Transfer Learning Model on Weather Images')
    plt.show()

    # Plot the training and validation accuracy
    plt.figure(1)
    plt.plot(train_acc.cpu(), label='Training accuracy')
    plt.plot(val_acc.cpu(), label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy for VGG16 Transfer Learning Model on Weather Images')
    plt.show()