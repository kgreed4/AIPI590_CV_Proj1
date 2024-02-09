import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


def initialize_model(num_classesg):
    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace the classifier of the VGG model
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model


if __name__ == "__main__":
    num_classes = 10
    model = initialize_model(num_classes)
    print(model)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)
