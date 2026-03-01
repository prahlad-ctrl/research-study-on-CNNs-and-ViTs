'''
making a default config file so that we can have somewhat get the closest term while comparing both the models 
( it wwont be fully accurate but this is the best we can do to try and have fair comparision)
'''

import torch

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed = 42
    batch_size = 128
    Lepochs = 30
    Mepochs = 60
    lr = 3e-4
    weight_decay = 0.05
    optimizer = "AdamW"
    
    # will add the num of images,res files, etc shortly
    
config = Config()