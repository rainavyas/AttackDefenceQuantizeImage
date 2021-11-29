'''
Use an uncertainty based detector to detect adversarial attacks
Evaluate (F1 score) for each uncertainty measure
'''

from uncertainty import ensemble_uncertainties_classification
import sys
import os
import argparse
import numpy as np
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
from models import Classifier
from data_prep import DataTensorLoader
from layer_handler import get_layer
from saliency_sub_attack_detect import get_fooling_rate, attack_imgs
from eval_sal_sub_residue_detector import get_best_f_score

if __name__ == '__main__':
     # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify target trained model')
    commandLineParser.add_argument('ARCH', type=str, help='vgg16, densenet121, resnet18, etc.')
    commandLineParser.add_argument('MODEL_BASE', type=str, help='Specify trained models base name')
    commandLineParser.add_argument('--num_seeds', type=int, default=5, help="Specify number of model seeds to evaluate with")
    commandLineParser.add_argument('--num_points', type=int, default=2000, help="number of data points to attack")
    args = commandLineParser.parse_args()

# Constant
    SIZE = 32
    NUM_CLASSES = 100
    device = torch.device('cpu')

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_sal_sub_uncertainty_detector.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load trained target model
    model = Classifier(args.ARCH, NUM_CLASSES, device, size=SIZE)
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Load the data as tensors and quantize
    dataloader = DataTensorLoader()
    imgs, labels = dataloader.get_test()
    imgs = imgs[:args.num_points]
    labels = labels[:args.num_points]
    imgs = dataloader.quantize(imgs, quantization=args.quantization)
    print("Loaded Data")

    # Perform saliency based substitution attack
    criterion = nn.CrossEntropyLoss().to(device)
    imgs_attacked = attack_imgs(imgs, labels, model, criterion, args.quantization, args.N, dataloader)
    print("Attacked images")

    # Evaluate impact of attack - keep successful attacks only
    _, fool_inds = get_fooling_rate(imgs, imgs_attacked, model, labels)
    imgs_original = imgs[fool_inds]
    imgs_attack = imgs_attacked[fool_inds]

    # Get prediction probabilities for all models
    original_probs = []
    attacked_probs = []
    for seed in range(1,args.num_seeds+1):
        model_path = f'{args.MODEL_BASE}_seed{seed}.th'
        model = Classifier(args.ARCH, NUM_CLASSES, device, size=SIZE)
        model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
        model.to(device)
        model.eval()

        with torch.no_grad():
            original_prob = softmax(model(imgs_original), axis=-1)
            attacked_prob = softmax(model(imgs_attack), axis=-1)
        original_probs.append(original_prob)
        attacked_probs.append(attacked_prob)

    original_probs = torch.stack(original_probs).cpu().detach().numpy()
    attacked_probs = torch.stack(attacked_probs).cpu().detach().numpy()

    # Get uncertainty measure for each data point
    original_uncertainties = ensemble_uncertainties_classification(original_probs)
    adv_uncertainties = ensemble_uncertainties_classification(attacked_probs)

    # For each uncertainty measure report F1-score
    labels = [0]*len(original_probs) + [1]*len(attacked_probs)

    for measure in original_uncertainties.keys():
        original = original_uncertainties[measure]
        adv = adv_uncertainties[measure]
        together = np.concatenate((original, adv))

        # Calculate best F1-score
        precision, recall, _ = precision_recall_curve(labels, together)
        best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
        print(f'{measure}: {best_f1}')