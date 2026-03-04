import numpy as np
import torch
import torch.nn.functional as F

class TrackMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.correct = 0
        self.total = 0
        self.confidences = []
        self.predictions = []
        self.labels = []
    
    def update(self, logits, targets, loss):
        self.losses.append(loss.item())
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
        
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)
        
        self.confidences.extend(confs.detach().cpu().numpy())
        self.predictions.extend(preds.detach().cpu().numpy())
        self.labels.extend(targets.detach().cpu().numpy())
    
    def t1_acc(self): # top 1 accuracy
        return 100.0* self.correct/self.total if self.total>0 else 0.0
    
    def loss_variance(self):
        return np.var(self.losses) if len(self.losses)>1 else 0.0
    
    def ec_error(self, n_bins= 10): # expected calibration error
        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        bin_boundaries = np.linspace(0, 1, n_bins+1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lowers, bin_uppers in zip(bin_lowers, bin_uppers):
            in_bin = (confidences>bin_lowers) & (confidences<= bin_uppers)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin>0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence = confidences[in_bin].mean()
                ece += np.abs(avg_confidence - accuracy_in_bin)* prop_in_bin
                
        return ece*100.0