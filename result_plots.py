import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_efficiency(subset_sizes, cnn_accuracies, vit_accuracies, save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)
    
    plt.plot(subset_sizes, cnn_accuracies, marker='o', linewidth=2, label='ResNet-18 (CNN)', color='#1f77b4')
    plt.plot(subset_sizes, vit_accuracies, marker='s', linewidth=2, label='ViT-Tiny (Transformer)', color='#ff7f0e')
    
    plt.xscale('log')
    plt.xlabel('Dataset Size (Log Scale)', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Sample Efficiency: Identifying the Dataset Size Threshold', fontsize=14)
    plt.axvline(x=10000, color='gray', linestyle='--', alpha=0.7, label='Hypothesized Crossover Point')
    
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    save_path = os.path.join(save_dir, 'efficiency.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_robustness_comparison(cnn_results_path, vit_results_path, save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with open(cnn_results_path, 'r') as f:
            cnn_data = json.load(f)["metrics"]
        with open(vit_results_path, 'r') as f:
            vit_data = json.load(f)["metrics"]
    except FileNotFoundError:
        print("no files found to plot")
        return

    corruptions = list(cnn_data.keys())
    cnn_accs = [cnn_data[c]["top1_accuracy"] for c in corruptions]
    vit_accs = [vit_data[c]["top1_accuracy"] for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width/2, cnn_accs, width, label='ResNet-18 (CNN)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, vit_accs, width, label='ViT-Tiny (Transformer)', color='#ff7f0e')

    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Robustness to Corruptions & Distribution Shifts', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in corruptions], rotation=15, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    save_path = os.path.join(save_dir, 'robustness_comparison.png')
    plt.savefig(save_path)
    plt.close()

def load_real_accuracies(model_name, subset_sizes, results_dir="results"):
    file_path = os.path.join(results_dir, f"{model_name}_accuracies.json")
    accuracies = []
        
    with open(file_path, "r") as f:
        data = json.load(f)
        
    for size in subset_sizes:
        acc = data.get(str(size), 0.0)
        accuracies.append(acc)
        
    return accuracies

if __name__ == "__main__":
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    real_cnn_acc = load_real_accuracies("resnet18", sizes)
    real_vit_acc = load_real_accuracies("vit_tiny", sizes)
    
    print(f"Real CNN Accuracies: {real_cnn_acc}")
    print(f"Real ViT Accuracies: {real_vit_acc}")
    
    plot_efficiency(sizes, real_cnn_acc, real_vit_acc)