import torch
import torch.nn as nn
from cnn_finetune import make_model
import torchvision.transforms as transforms

class Classifier(nn.Module):
    def __init__(self, arch, num_classes, device=torch.device('cpu'), size=32) -> None:
        super().__init__()
        self.main_model =  make_model(arch, num_classes=num_classes, pretrained=True, input_size=(size, size))

        self.means = torch.FloatTensor([0.5071, 0.4865, 0.4409]).view(1,-1,1,1).to(device)
        self.stds = torch.FloatTensor([0.2009, 0.1984, 0.2023]).view(1,-1,1,1).to(device)
        self.largest = 255
    
    def forward(self, X):
        '''
        X: torch Tensor
                [B x C x H x W]
        '''
        X_norm = ((X/self.largest) - self.means)/self.stds
        return self.main_model(X_norm)

