import argparse
import torch
from os.path import isdir
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Argument Parser
def arg_parser():
    parser = argparse.ArgumentParser(description="Train an Image Classifier")
    parser.add_argument('--arch', default="vgg16", type=str, help="Model architecture (default: vgg16)")
    parser.add_argument('--save_dir', default="./checkpoint.pth", help="Directory to save checkpoints")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--hidden_units', type=int, default=120, help="Number of hidden units in classifier")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--gpu', action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

# Data Transforms
def create_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Load Data
def load_data(data_dir, train=True):
    transforms = create_transforms(train)
    dataset = datasets.ImageFolder(data_dir, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=train)
    return dataset, loader

# Select Device
def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        if use_gpu:
            print("CUDA not available. Using CPU instead.")
        return torch.device("cpu")

# Load Pretrained Model
def load_pretrained_model(arch="vgg16"):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    for param in model.parameters():
        param.requires_grad = False
    return model

# Build Classifier
def build_classifier(input_size, hidden_units):
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

# Validate Model
def validate_model(model, loader, criterion, device):
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            probs = torch.exp(outputs)
            equality = (labels == probs.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean().item()
    return test_loss / len(loader), accuracy / len(loader)

# Train Model
def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, print_every):
    steps = 0
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss:.3f}.. "
                      f"Validation Accuracy: {valid_accuracy:.3f}")
                running_loss = 0
                model.train()

# Save Checkpoint
def save_checkpoint(model, save_dir, train_data):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)
    print(f"Checkpoint saved to {save_dir}")

# Main Function
def main():
    args = arg_parser()
    device = get_device(args.gpu)

    # Data directories
    data_dir = "flowers"
    train_dir, valid_dir, test_dir = f"{data_dir}/train", f"{data_dir}/valid", f"{data_dir}/test"

    # Load data
    train_data, train_loader = load_data(train_dir, train=True)
    valid_data, valid_loader = load_data(valid_dir, train=False)
    _, test_loader = load_data(test_dir, train=False)

    # Load model
    model = load_pretrained_model(args.arch)
    model.name = args.arch
    model.classifier = build_classifier(25088, args.hidden_units)
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, print_every=30)

    # Validate on test data
    test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.3f}.. Test Accuracy: {test_accuracy:.3f}")

    # Save the model
    save_checkpoint(model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
