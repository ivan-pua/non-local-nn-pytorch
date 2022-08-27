'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
import argparse

from models.resnet2D import resnet2D56

def main(args):  

    if args.verbose:
        from utils import progress_bar

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 30

    # Data
    print('==> Preparing data..')

    img_size = 128
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    trainset = torchvision.datasets.ImageFolder(root='../ChestXRays/train2', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='../ChestXRays/val2', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)

    classes = ('covid', 'normal', 'pneumonia', 'tuberculosis')

    # Model
    print('==> Building model..')
    if args.nl:
        print("ResNet-56 with non-local block after second residual block..")
        net = resnet2D56(non_local=True)
    else:
        print("ResNet-56 without non-local block..")
        net = resnet2D56(non_local=False)



    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        global checkpoint, best_acc, start_epoch

        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=start_epoch - 1)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # print(np.array(inputs).shape)
            inputs, targets = inputs, targets
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if not args.verbose:
            print('Loss: %.3f' % train_loss)

        return train_loss

    def test(epoch):

        y_pred, y_true = [],[]

        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)

                # Confusion Matrix
                output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                y_pred.extend(output) # Save Prediction
                loss = criterion(outputs, targets)

                labels = targets.data.cpu().numpy()
                y_true.extend(labels) # Save Truth

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if args.verbose:
                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)

        if not args.verbose:
            print('Loss: %.3f' % test_loss)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc
        return test_loss

    tr_loss_list = []
    tst_loss_list = []

    for epoch in range(start_epoch, start_epoch+num_epochs):
        train_l = train(epoch)
        lr_scheduler.step()
        test_l = test(epoch)
        tr_loss_list.append(train_l)
        tst_loss_list.append(test_l)

    print("Best Accuracy: ", best_acc)
    print("-----------------------------------------------")

    print("train loss")
    print(tr_loss_list)
    print("test loss")
    print(tst_loss_list)

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--verbose', '-v', action='store_true', help='display progress bar')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--nl', '-n', action='store_true', help='add non-local block')
    args = parser.parse_args()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    main(args)