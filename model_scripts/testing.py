import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from dataloading import setup_dataloaders
from model_training import initialize_model

"""
    Function to run the model on the test data, calculate evaluation metrics,
    and visualize performance.

    Parameters:
        model: PyTorch model to be tested.
        test_dl: DataLoader containing test data.
        pathtocsv: Path to the CSV file containing test data labels.
        nameOfModel: Name of the model to be used in file name for test results.
    """
def test_model(model_path, test_loader, model_name):

    # Put the model in evaluation mode
    model.eval()

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    pred_labels = []
    pred_probs = []

    # Unmap numerical labels to original class names
    class_to_label = {
        0: 'dew',
        1: 'fogsmog',
        2: 'frost',
        3: 'glaze',
        4: 'hail',
        5: 'lightning',
        6: 'rain',
        7: 'rainbow',
        8: 'rime',
        9: 'sandstorm',
        10: 'snow'
    }

    # Iterate through the test data and make predictions
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            softmax_probs = torch.nn.functional.softmax(outputs, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())  # Use the predicted labels directly
        pred_probs.extend(softmax_probs.cpu().numpy())


    print('pred labels 1', pred_labels)

    # Unmap numerical labels to original class names
    true_labels = [class_to_label[label] for label in true_labels]
    print(true_labels)
    for label in pred_labels:
    #   print('preds class', class_to_label)
        print('preds label', class_to_label[label])
    pred_labels = [class_to_label[label] for label in pred_labels]

    # Convert true and predicted labels to one-hot encoding
    true_labels_onehot = label_binarize(true_labels, classes=class_to_label.values())
    pred_probs = np.array(pred_probs)

    # Compute overall accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Write evaluation metrics to a text file
    evaluation_file = f"{model_name}_evaluation.txt"
    with open(evaluation_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")

    # Print evaluation metrics
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Compute Precision-Recall curve and ROC curve per class
    precision = dict()
    recall = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pr_auc = dict()

    for i, class_name in enumerate(class_to_label.values()):
        class_index = list(class_to_label.values()).index(class_name)
        precision[class_name], recall[class_name], _ = precision_recall_curve(true_labels_onehot[:, class_index], pred_probs[:, class_index])
        fpr[class_name], tpr[class_name], _ = roc_curve(true_labels_onehot[:, class_index], pred_probs[:, class_index])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
        pr_auc[class_name] = auc(recall[class_name], precision[class_name])

    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    plt.title('Precision-Recall Curve per Class')
    for class_name in class_to_label.values():
        plt.plot(recall[class_name], precision[class_name], label=f'{class_name} (AUC = {pr_auc[class_name]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"model_results/{model_name}_precision_recall_curve.png")
    plt.show()

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.title('ROC Curve per Class')
    for class_name in class_to_label.values():
        plt.plot(fpr[class_name], tpr[class_name], label=f'{class_name} (AUC = {roc_auc[class_name]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"model_results/{model_name}_roc_curve.png")
    plt.show()

    # Compute and plot confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(class_to_label.values()))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_to_label))
    plt.xticks(tick_marks, class_to_label.values(), rotation=45)
    plt.yticks(tick_marks, class_to_label.values())
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f"model_results/{model_name}_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    # Set your file path
    filepath = './models_saved/'

    # Load the model, specifying map_location to load it on GPU
    state_dict = torch.load(os.path.join(filepath, 'Resnet18_Transfer_Learning_0001_26.pth'), map_location=torch.device('cuda'))

    # Instance
    # model = models.resnet18(pretrained=True)
    model = initialize_model(11, model_name="resnet18")
    model.load_state_dict(state_dict, strict=False)

    # set up data loaders
    train_loader, val_loader, test_loader = setup_dataloaders(balance_classes=False, augment=False)
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}")

    # Move the test data to the GPU
    test_data = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        test_data.append((data, target))

    test_model(model, test_data, 'Resnet18_TL_0001')