import argparse
import json
from math import ceil
from PIL import Image
import torch
import numpy as np
from torchvision import models

def arg_parser():
    """Parse command-line arguments for predict.py."""
    parser = argparse.ArgumentParser(description="Image Classifier Prediction Script")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file.', required=True)
    parser.add_argument('--top_k', type=int, help='Number of top predictions to return.', default=5)
    parser.add_argument('--category_names', type=str, help='Path to category-to-name JSON file.', default='cat_to_name.json')
    parser.add_argument('--gpu', action="store_true", help='Use GPU for inference if available.')

    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    """Load model checkpoint and rebuild the model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Ensure compatibility with CPU/GPU environments

    if checkpoint['structure'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported model structure: {checkpoint['structure']}")

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    return model

def process_image(image_path):
    """Process an image for use with a PyTorch model."""
    img = Image.open(image_path)

    # Resize the image while maintaining aspect ratio
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:
        img = img.resize((int(aspect_ratio * 256), 256))
    else:
        img = img.resize((256, int(256 / aspect_ratio)))

    # Center crop to 224x224
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Convert to numpy array and normalize
    numpy_img = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_img = (numpy_img - mean) / std

    # Reorder dimensions for PyTorch (C x H x W)
    return numpy_img.transpose((2, 0, 1))

def predict(image_path, model, cat_to_name, top_k=5, device='cpu'):
    """Predict the class of an image using a trained model."""
    # Move model to the specified device
    model.to(device)
    model.eval()

    # Process image and prepare it as a tensor
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor).to(device)

    # Forward pass through the model to get predictions
    with torch.no_grad():
        log_probs = model.forward(img_tensor)

    # Convert log probabilities to linear scale
    linear_probs = torch.exp(log_probs)

    # Get the top K predictions
    top_probs, top_indices = linear_probs.topk(top_k)

    # Convert to lists and map indices to class labels and flower names
    top_probs = top_probs.cpu().numpy().squeeze().tolist()
    top_indices = top_indices.cpu().numpy().squeeze().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    top_flowers = [cat_to_name[class_] for class_ in top_classes]

    return top_probs, top_classes, top_flowers

def print_probabilities(probs, flowers):
    """Print the top predictions along with their probabilities."""
    print("\nTop Predictions:")
    for i, (flower, prob) in enumerate(zip(flowers, probs)):
        print(f"Rank {i+1}:")
        print(f"  Flower: {flower}")
        print(f"  Likelihood: {ceil(prob * 100)}%\n")

def main():
    """Main function for the prediction script."""
    args = arg_parser()

    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)

    # Set the device (CPU/GPU)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    # Make predictions
    probs, _, flowers = predict(args.image, model, cat_to_name, top_k=args.top_k, device=device)

    # Print the probabilities
    print_probabilities(probs, flowers)

if __name__ == "__main__":
    main()
