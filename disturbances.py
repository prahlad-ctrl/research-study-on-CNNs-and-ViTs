import torch
import torchvision.transforms as transforms
import random

# shuffling patches of image distrupts the global structure while keeping local texture
# vit is generally good at global shape while cnn is good at local texture, so this should be more challenging for vit than cnn
class PatchShuffle:
    def __init__(self, num_patches = 4):
        self.num_patches = num_patches
    
    def __call__(self, img):
        _, h, w = img.size()
        patch_h, patch_w = h//self.num_patches, w//self.num_patches
        
        patches = []
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                patches.append(img[:, i*patch_h:(i+1)* patch_h, j* patch_w:(j+1)* patch_w])
                # slicing img[channel, height, width]
        random.shuffle(patches)
        
        # reconstruct
        rows = [torch.cat(patches[i*self.num_patches:(i+1)*self.num_patches], dim=2) for i in range(self.num_patches)]
        shuffled_img = torch.cat(rows, dim=1)
        
        return shuffled_img

# evaluating under disturbances like noise, blur, etc
def get_dist_transforms(img_size, severity = 1):
    base_resize = transforms.Compose([transforms.Resize((img_size, img_size)),
                                      transforms.ToTensor()])
    dist = {'clean': transforms.Compose([base_resize,
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
            'gaussian_noise': transforms.Compose([base_resize,
                                                  transforms.Lambda(lambda x: x+torch.randn_like(x)* (0.1* severity)),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
            "gaussian_blur": transforms.Compose([transforms.Resize((img_size, img_size)),
                                                 transforms.GaussianBlur(kernel_size=5, sigma=(0.1 + 0.5* severity)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
            "texture_shift": transforms.Compose([transforms.Resize((img_size, img_size)),
                                                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1*severity),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
            "shuffled_patches": transforms.Compose([base_resize,
                                                    PatchShuffle(num_patches=4),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])}
    return dist