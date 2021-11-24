# Objective
Images are typically stored in 3 channels (RGB), with each channel having 256 unique values (0-255). Here, CIFAR-100 images are quantized to a desired quantization value (<256) and these images are then used to train state of the art deep learning classifiers. In this work, we want to perform adversarial attacks on quantized images, i.e. attacks in highly discrete image input spaces. 

Adversarial attacks of highly discrete (large amount of quantization) will mimic saliency based substitution adversarial attacks in the Natural Language Processing domain. Specifically, the $N$ most salient pixels will be identified in an image (salient wrt to classification training loss) and will be substituted for the closest quantized pixel that maximises the loss.  

For less discretized image inputs, attacks will be carried out in the standard continuous Projected Gradient Descent approach.

The impact on adversarial attack detection systems (specifically residue detection, uncertainty and Mahalanobis Distance) can then be determined.

# Requirements

Minimum python3.7

## Install with PyPI

`pip install datasets`

`pip install torch, torchvision`

`pip install cnn_finetune`

`pip install numpy`
