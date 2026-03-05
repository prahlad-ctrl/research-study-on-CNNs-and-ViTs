import torch
from torch import nn, optim
from tqdm import tqdm

from config import config
from dataset import build_dataloaders
from models import model
from metrics import TrackMetrics

import argparse
import json
import os

def train_o(model, dataloader, criterion, optimizer, device, tracker):
    model.train()
    tracker.reset()
    
    pbar = tqdm(dataloader)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        tracker.update(outputs, labels, loss)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{tracker.t1_acc():.2f}%"})

def evaluate(model, dataloader, criterion, device, tracker):
    model.eval()
    tracker.reset()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            tracker.update(outputs, labels, loss)
            
    return tracker.t1_acc(), tracker.ec_error()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vit_tiny', 'vit_small']) # bash to train all
    parser.add_argument('--dataset_size', type=int, default=10000, choices=[1000, 5000, 10000, 50000, 100000])
    parser.add_argument('--img_size', type=int, default=64, choices=[64, 128, 256, 512])
    parser.add_argument('--pretrained', action='store_true')
    args, unknown = parser.parse_known_args()
    # even though i made these many options, ill not train every single one ofcourse

    print(f"model: {args.model} and data size: {args.dataset_size}") # will be training all models and changing some data so we have to know

    trainloader, testloader = build_dataloaders(
        dataset_name="cifar10", 
        subset_size=args.dataset_size, 
        img_size=args.img_size
    )

    model = model(
        model_name=args.model, 
        num_classes=10, 
        pretrained=args.pretrained, 
        img_size=args.img_size
    )
    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.Lepochs)
    
    train_tracker = TrackMetrics()
    test_tracker = TrackMetrics()

    for epoch in range(config.Lepochs):
        print(f"\nepoch {epoch+1}/{config.Lepochs}")
        
        train_o(model, trainloader, criterion, optimizer, config.device, train_tracker)
        scheduler.step()
        
        train_loss_var = train_tracker.loss_variance()
        
        val_acc, val_ece = evaluate(model, testloader, criterion, config.device, test_tracker)
        print(f"val acc: {val_acc:.2f}% , val ece: {val_ece:.2f} , loss variance: {train_loss_var:.4f}")

    os.makedirs("trained_mods", exist_ok=True)
    torch.save(model.state_dict(), f"trained_mods/{args.model}.pth")
    
    os.makedirs("results", exist_ok=True)
    results_file = f"results/{args.model}_training.json"
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            training_data = json.load(f)
    else:
        training_data = {}

    training_data[str(args.dataset_size)] = val_acc
    
    with open(results_file, "w") as f:
        json.dump(training_data, f, indent=4)