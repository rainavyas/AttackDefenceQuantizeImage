'''
1) Saliency ranked substitution attack on discrete image pixels - on train data
2) Train Residue Detector - save trained detector
'''

import sys
import os
import argparse
import torch
import torch.nn as nn
from tools import AverageMeter, accuracy_topk, get_default_device
from models import Classifier
from data_prep import DataTensorLoader
from layer_handler import get_layer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def get_fooling_rate(X, X_attack, model, labels):
    '''
    X: torch tensor [B x C x H x W]
    Report adversarial attack fooling rate
    Fooling rate only includes originally correctly classified points
    Return indices of sucessful attacks and fooling rate
    '''
    model.eval()
    with torch.no_grad():
        logits_original = model(X)
        pred_original = torch.argmax(logits_original, dim=-1)
        logits_attack = model(X_attack)
        pred_attack = torch.argmax(logits_attack, dim=-1)
    
    total_count = 0
    fool_inds = []

    for j, (o, a, l) in enumerate(zip(pred_original, pred_attack, labels)):
        if o != l:
            continue
        total_count += 1
        if a != o:
            fool_inds.append(j)
    
    fool_rate = len(fool_inds)/total_count
    print("Total Count: ", total_count)
    print("Fool Rate: ", fool_rate)
    assert(len(fool_inds)!=0)
    return fool_rate, fool_inds


class LayerClassifier(nn.Module):
    '''
    Simple Linear classifier
    '''
    def __init__(self, dim, classes=2):
        super().__init__()
        self.layer = nn.Linear(dim, classes)
    def forward(self, X):
        return self.layer(X)

def train(train_loader, model, criterion, optimizer, epoch, device, out_file, print_freq=1):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            text = '\n Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, prec=accs)
            print(text)
            with open(out_file, 'a') as f:
                f.write(text)

def eval(val_loader, model, criterion, device, out_file):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

    text ='\n Test\t Loss ({loss.avg:.4f})\t Accuracy ({prec.avg:.3f})\n'.format(loss=losses, prec=accs)
    print(text)
    with open(out_file, 'a') as f:
        f.write(text)

def attack_imgs(X, labels, model, criterion, quantization, N, dataloader):
    '''
    Saliency based pixel substitution attack
    Substition is only with closest quantized value (in saliency direction)

    Inputs:
        X: torch tensor [B x C x H x W]
        labels: torch tensor [B]
        quantization: pixel quantization permitted (0-255)
        N: Number of pixels to substitute
    
    Outputs:
        X_attacked: torch tensor [B x C x H x W]
    '''

    # Get saliency per pixel
    X.requires_grad=True
    X.retain_grad()
    logits = model(X)
    loss = criterion(logits, labels)
    loss.backward()
    X_grads = X.grad
    assert(X.grad is not None)
    X_saliencies = torch.abs(X_grads)

    X_attacks= []
    LARGEST = 256
    step_size = (LARGEST/quantization-1)

    for j in range(X.size(0)):
        curr_sal = X_saliencies[j]
        curr_X = X[j]
        curr_grad = X_grads[j]

        # Create mask to keep top N saliencies
        nth_largest, _ = torch.topk(torch.reshape(curr_sal, (-1,)), N)
        nth_largest = nth_largest[-1]
        curr_grad[curr_sal<nth_largest] = 0

        # Apply substitution
        X_sub = curr_X + (step_size*torch.sign(curr_grad))
        X_attacked = dataloader.quantize(X_sub, quantization=quantization)
        X_attacks.append(X_attacked)

    return torch.stack(X_attacks)

if __name__ == '__main__':
     # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('LOG', type=str, help='Output file to log activities of this script')
    commandLineParser.add_argument('MODEL', type=str, help='Specify trained model th file')
    commandLineParser.add_argument('ARCH', type=str, help='vgg16, densenet121, resnet18, etc.')
    commandLineParser.add_argument('RESIDUE', type=str, help='Specify trained residue detector out path')
    commandLineParser.add_argument('--quantization', type=int, default=256, help="Specify quantization")
    commandLineParser.add_argument('--N', type=int, default=600, help="Specify number of pixels to substitute")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    commandLineParser.add_argument('--B', type=int, default=100, help="Specify residue training batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify residue training epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify residue training learning rate")
    commandLineParser.add_argument('--num_points', type=int, default=2000, help="number of data points to attack")

    args = commandLineParser.parse_args()
    torch.manual_seed(args.seed)

    # Constant
    SIZE = 32
    NUM_CLASSES = 100

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/saliency_sub_attack_detect.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if args.cpu == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()
    
    # Load trained model
    model = Classifier(args.ARCH, NUM_CLASSES, device, size=SIZE)
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Load the data as tensors and quantize
    dataloader = DataTensorLoader()
    imgs, labels = dataloader.get_test()
    # imgs, labels = dataloader.get_train()
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
    X_original = get_layer(imgs_original, model, args.ARCH)
    X_attack = get_layer(img_attack, model, args.ARCH)

    # Train residue detector with successful attacks
    targets = torch.LongTensor([0]*len(X_original)+[1]*len(X_attack))
    X = torch.cat((X_original, X_attack))
    indices = torch.randperm(len(targets))
    targets = targets[indices]
    X = X[indices]
    ds = TensorDataset(X, targets)
    dl = DataLoader(ds, batch_size=args.B, shuffle=True)

    # Model
    detector = LayerClassifier(X.size(-1))
    detector.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Create file
    with open(args.LOG, 'w') as f:
        text = f' Detector trained using model {args.MODEL}, N {args.N}\n'
        f.write(text)

    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        text = '\n current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        with open(args.LOG, 'a') as f:
            f.write(text)
        print(text)
        train(dl, detector, criterion, optimizer, epoch, device, args.LOG)
    
    # Save the trained detector model for identifying adversarial attacks
    torch.save(detector.state_dict(), args.DETECTOR)

