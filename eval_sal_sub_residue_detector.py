'''
Evaluate performance of residue detector

1) Attack quantized test data using saliency ranked substitution attack
2) Evaluate (F1 score) trained residue detector
'''

import sys
import os
import argparse
import torch
import torch.nn as nn
from models import Classifier
from data_prep import DataTensorLoader
from layer_handler import get_layer
from saliency_sub_attack_detect import get_fooling_rate, LayerClassifier, attack_imgs
import numpy as np
from sklearn.metrics import precision_recall_curve

def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precisions*(beta**2))+recalls))
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

if __name__ == '__main__':
     # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify trained model th file')
    commandLineParser.add_argument('ARCH', type=str, help='vgg16, densenet121, resnet18, etc.')
    commandLineParser.add_argument('RESIDUE', type=str, help='Specify trained residue detector path')
    commandLineParser.add_argument('--quantization', type=int, default=256, help="Specify quantization")
    commandLineParser.add_argument('--N', type=int, default=600, help="Specify number of pixels to substitute")
    commandLineParser.add_argument('--num_points', type=int, default=2000, help="number of data points to attack")
    args = commandLineParser.parse_args()

    # Constant
    SIZE = 32
    NUM_CLASSES = 100
    device = torch.device('cpu')

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_sal_sub_residue.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load trained model
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
    img_attack = imgs_attacked[fool_inds]

    # Map to encoder embedding space
    with torch.no_grad():
        model.eval()
        X_original = get_layer(imgs_original, model, args.ARCH)
        X_attack = get_layer(img_attack, model, args.ARCH)
    targets = torch.LongTensor([0]*len(X_original)+[1]*len(X_attack))
    X = torch.cat((X_original, X_attack))

    # Load trained residue detector
    detector = LayerClassifier(X.size(-1))
    detector.load_state_dict(torch.load(args.RESIDUE, map_location=torch.device('cpu')))
    detector.to(device)
    detector.eval()

    # get predicted logits of being adversarial attack
    with torch.no_grad():
        logits = detector(X)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        adv_probs = probs[:,1].squeeze().cpu().detach().numpy()
    
    print("Got prediction probs")
    # get precision recall values and highest F1 score (with associated prec and rec)
    precision, recall, _ = precision_recall_curve(labels, adv_probs)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print(f'Precision: {best_precision}\t Recall: {best_recall}\t F1: {best_f1}')

    # # plot all the data
    # plt.plot(recall, precision, 'r-')
    # plt.plot(best_recall,best_precision,'bo')
    # plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.savefig(out_file)