import torch
import torch.nn as nn


def get_layer(X, trained_model, arch='vgg16'):
    '''
    Return encoder embedding space features
    Only designed for vgg16 so far
    '''
    if arch == 'vgg16':
        return get_vgg_layer(X, trained_model)
    else:
        raise ValueError("Invalid architecture")

def get_vgg_layer(X, trained_model):
  '''
  Embedding space chosen is the space after all VGG layers and
  before the classifier stage

  Pass through one linear layer of classifier too
  '''
  features = trained_model.features(X)
  part_classifier = nn.Sequential(*list(trained_model._classifier.children())[:3])
  return part_classifier(features.squeeze())