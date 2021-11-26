'''
Evaluate performance of residue detector

1) Attack quantized test data using saliency ranked substitution attack
2) Evaluate (F1 score) using Mahalanobis distance detector
'''

import sys
import os
import argparse
import torch
import torch.nn as nn
from models import Classifier
from data_prep import DataTensorLoader
from saliency_sub_attack_detect import get_fooling_rate, attack_imgs
from eval_sal_sub_residue_detector import get_best_f_score
import numpy as np
from sklearn.metrics import precision_recall_curve


def calculate_per_class_dist(vector, class_mean, inv_cov):
    diff = vector - class_mean
    print(diff)
    half = np.matmul(inv_cov, diff)
    return np.dot(diff, half)

def calculate_mahalanobis(vector, class_means, inv_cov):
    # Select closest class conditional distance
    dists = []
    for class_mean in class_means:
        dists.append(calculate_per_class_dist(vector, class_mean, inv_cov))
    return min(dists)

if __name__ == '__main__':
     # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify trained model th file')
    commandLineParser.add_argument('ARCH', type=str, help='vgg16, densenet121, resnet18, etc.')
    commandLineParser.add_argument('--quantization', type=int, default=256, help="Specify quantization")
    commandLineParser.add_argument('--N', type=int, default=600, help="Specify number of pixels to substitute")
    commandLineParser.add_argument('--num_points_test', type=int, default=2000, help="number of data points to attack")
    commandLineParser.add_argument('--num_points_train', type=int, default=2000, help="number of data points to use for mahalanobis train")
    args = commandLineParser.parse_args()

    # Constant
    SIZE = 32
    NUM_CLASSES = 100
    device = torch.device('cpu')

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_sal_sub_mahalanobis_detector.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load trained model
    model = Classifier(args.ARCH, NUM_CLASSES, device, size=SIZE)
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Load train data as tensors and quantize
    dataloader = DataTensorLoader()
    imgs, labels = dataloader.get_test()
    imgs = imgs[:args.num_points_train]
    labels = labels[:args.num_points_train]
    imgs = dataloader.quantize(imgs, quantization=args.quantization)

    # Obtain output train logits
    with torch.no_grad():
        logits = model(imgs)
    
    # Group logits by class to calculate class means and joint cov
    logits_by_class_list = []
    for i in range(NUM_CLASSES):
        logits_by_class_list.append([])
    
    for logit, l in zip(logits, labels):
        logits_by_class_list[l].append(logit.squeeze())
    
    class_means = []
    cov = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        class_logits = torch.stack(logits_by_class_list[i]).cpu().detach().numpy()
        class_mean = np.mean(class_logits, axis=0)
        class_means.append(class_mean)

        class_cov = np.cov(class_logits, rowvar=False)
        cov += class_cov
    cov = cov/NUM_CLASSES
    inv_cov = np.linalg.inv(cov)

    # Load the test data as tensors and quantize
    imgs, labels = dataloader.get_test()
    imgs = imgs[:args.num_points_test]
    labels = labels[:args.num_points_test]
    imgs = dataloader.quantize(imgs, quantization=args.quantization)
    print("Loaded Test Data")

    # Perform saliency based substitution attack
    criterion = nn.CrossEntropyLoss().to(device)
    imgs_attacked = attack_imgs(imgs, labels, model, criterion, args.quantization, args.N, dataloader)
    print("Attacked images")

    # Evaluate impact of attack - keep successful attacks only
    _, fool_inds = get_fooling_rate(imgs, imgs_attacked, model, labels)
    imgs_original = imgs[fool_inds]
    imgs_attack = imgs_attacked[fool_inds]

    # Map to logit space
    with torch.no_grad():
        model.eval()
        X_original = model(imgs_original)
        X_attack = model(imgs_attack)
    targets = torch.LongTensor([0]*len(X_original)+[1]*len(X_attack))
    X = torch.cat((X_original, X_attack))

    # Calculate Mahalanobis distances
    dists = [calculate_mahalanobis(v, class_means, inv_cov) for v in X]

    # get precision recall values and highest F1 score (with associated prec and rec)
    precision, recall, _ = precision_recall_curve(targets, dists)
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)
    print(f'Precision: {best_precision}\t Recall: {best_recall}\t F1: {best_f1}')

    # # plot all the data
    # plt.plot(recall, precision, 'r-')
    # plt.plot(best_recall,best_precision,'bo')
    # plt.annotate(F"F1={best_f1:.2f}", (best_recall,best_precision))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.savefig(out_file)