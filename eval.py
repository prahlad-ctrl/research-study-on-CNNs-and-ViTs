import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import argparse
import json
import os
from tqdm import tqdm

from config import config
from models import model
from disturbances import get_transforms
from metrics import TrackMetrics

def eval_robustness(model, dataloader, criterion, device):
    model.eval()
    tracker = TrackMetrics()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            tracker.update(outputs, labels, loss)
            
    return tracker.t1_acc(), tracker.ec_error()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet50', 'vit_tiny', 'vit_small'])
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()
    
    model = model(model_name=args.model, num_classes=10, pretrained=False, img_size=args.img_size)
    model.load_state_dict(torch.load(args.weights, map_location=config.device))
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    disturbance = get_transforms(img_size=args.img_size, severity=2)
    results = {
        "model": args.model,
        "resolution": args.img_size,
        "metrics": {}
    }

    for dist_name, transform in disturbance.items():
        print(f"\nevaluating on: {dist_name.upper()}")
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=4)
        
        acc, ece = eval_robustness(model, testloader, criterion, config.device)
        print(f"accuracy: {acc:.2f}% , expected calibration error: {ece:.2f}")
        
        results["metrics"][dist_name] = {"top1_accuracy": acc,
                                         "calibration_error": ece}

    os.makedirs("results", exist_ok=True)
    save_path = f"results/{args.model}_eval.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()