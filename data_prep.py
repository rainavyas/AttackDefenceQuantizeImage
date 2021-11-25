'''
Prepare the CIFAR-100 dataset as tokenized torch tensors
Functionality to quantize the image data to desired quantization
'''

import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np


class DataTensorLoader():
    def __init__(self):
        self.dataset = load_dataset('cifar100')
    
    def _get_data(self, data):

        imgs = data['img']
        labels = data['fine_label']

        labels = torch.LongTensor(labels)
        imgs = torch.FloatTensor(imgs)
        imgs = torch.transpose(imgs, 1, 3)
        imgs = torch.transpose(imgs, 2, 3)

        return  imgs, labels

    def get_train(self):
        return self._get_data(self.dataset['train'])

    def get_test(self):
        return self._get_data(self.dataset['test'])
    
    def quantize(self, X, quantization=256):
        '''
        Quantize data
        Input X: torch tensor [B x C x H x W]
        '''
        bins = np.linspace(0, 255, num=quantization)
        centres = centres = (bins[1:]+bins[:-1])/2

        X = X.cpu().detach().numpy()
        inds = np.digitize(X, centres)
        X_quantized = torch.FloatTensor(np.vectorize(lambda i: bins[i])(inds.astype(int)))
        return X_quantized


