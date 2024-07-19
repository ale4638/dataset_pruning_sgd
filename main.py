

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from IPython import embed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import argparse
import model
from tqdm import tqdm
import numpy as np
from utils import progress_bar
from utils import *
from models import *
from utils import model_train as train
from utils import model_test as test
import cvxpy as cp
from torch.utils.data import DataLoader
from calc_influence_function import calc_s_test_single_icml, calc_s_test_sgd
import scipy.io as scio
import joblib
import copy
import gc
def dataset_pruning(output_path,model,data,dataset='cifar10'):
    influence_score_list = []
    features,labels = data
    #get all_data
 
    global all_data
    
    # for i in tqdm(range(len(features))):
    #     data,label = features[i].to(device), torch.tensor(labels[i]).to(device)
    #     #using liang's method
    #     influence_score = calc_s_test_single_icml(model,data,label,all_data,gpu=0,recursion_depth=1, r=1)
    #     #appling sgd
    #     # incluence_score = calc_s_test_single_sgd(model,data,label,all_data,gpu=0,recursion_depth=1, r=1)
        
        # influence_score_list.append(influence_score)
    influence_score_list =  calc_s_test_sgd(model,all_data, gpu=0,
                       damp=0.01, scale=25,r=1,target_epoch = 1)
    np.save('./checkpoint/influence_score_%s.npy'%dataset,influence_score_list)
    return influence_score_list

def pruned_dataset_training(epoch):
    score_list = np.load('./influence_score_list.npy')
    temp = score_list.copy()
    temp.sort()
    threshold = temp[10000]
    index = score_list >=threshold
    pruned_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    pruned_dataset.data = pruned_dataset.data[index]
    pruned_dataset.targets = list(np.array(pruned_dataset.targets)[index])
    pruned_loader = torch.utils.data.DataLoader(
    pruned_dataset, batch_size=128, shuffle=True, num_workers=2)
    train(epoch,pruned_loader)



def extract_feature(net,dataloader,dataset='cifar10'):
    # checkpoint = torch.load('./checkpoint/ckpt_%s.pth'%dataset)
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print('Pretrained model acc: %.3f \n' % best_acc)
    feature_list = []
    label_list = []
    net.eval()
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        # compute output
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs,_ = net(inputs,out_feature=True)
        feature_list.append(outputs.squeeze().cpu().data)
        label_list.append(labels.item())
    torch.save([feature_list,label_list],'checkpoint/%s_features.pt'%dataset)
    return [feature_list,label_list]


def m_guided_opt(S,size):
    n,m = S.shape
    W = cp.Variable(n, boolean=True)
    constaints = [cp.sum(W)==size]
    obj = cp.Minimize(cp.norm(W@S,2))
    prob = cp.Problem(obj, constaints)
    prob.solve(solver=cp.CPLEX, verbose = True)
    W_optim = W.value
 
    return W_optim

def epsilon_guided_opt(S,epsilon):
    n,m = S.shape
    W = cp.Variable(n, boolean=True)
    constaints = [cp.norm(W@S,2)<=epsilon]
    obj = cp.Maximize(cp.sum(W))
    prob = cp.Problem(obj, constaints)
    prob.solve(solver=cp.CPLEX)
    W_optim = W.value
 
    return W_optim


if __name__ == '__main__':
    
    
    
    #test
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10/cifar100')
    parser.add_argument('--model', default='resnet18', type=str, help='model')
    parser.add_argument('--m', default=0, type=int, help='pruning size')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    
    # #get all_data
    # all_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(),download=True)
    # data = []
    # labels = []

    # for i in range(50000):
    #     data.append(all_data[i][0])
    #     labels.append(all_data[i][1])
    # all_data = [data,labels]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)

    wholedataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)

    wholeloader = torch.utils.data.DataLoader(
        wholedataset, batch_size=1, shuffle=False, num_workers=1)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = model.ResNet18()
    net = net.to(device)
    # print(net)

    # Pre-train
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    #for storing 
    bundle_size = 200
    list_of_nets = [net.to(device) for _ in range(bundle_size)]
    output_path = "temp"
    for epoch in range(0, 2):
        #train
        
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        info = []
        c = 0
        k = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #save
            list_of_nets[c].load_state_dict(copy.deepcopy(net.state_dict()))
            info.append({'idx': batch_idx,'train_points':(inputs,targets),'lr' : args.lr})
            c += 1
            if c == bundle_size or batch_idx == len(trainloader) - 1:
                fn = '%s/epoch%02d_bundled_models%02d.dat' % (output_path, epoch, k)
                models = model.NetList(list_of_nets)
                torch.save(models.state_dict(), fn)
                k+=1
                c=0
            #sgd
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #save
        
        fn = '%s/epoch%02d_info.dat' % (output_path, epoch)
        joblib.dump(info, fn, compress=9)
        #test
        test(epoch,testloader,net)
        scheduler.step()

    # Extract feature
    features,labels = extract_feature(net,wholeloader,dataset=args.dataset)
    all_data = [features,labels]
    
    
    if args.dataset=='cifar10':
        classifier = nn.Linear(50,10).cuda()
        classifier.load_state_dict(net.linear2.state_dict())

    correct = 0
    total = 0
#     for i in range(len(features)):
#         print('a')
#         output = classifier(features[i].cuda())
#         print('b')
#         label = labels[i]
#         _, predicted = output.max(0)

#         total += 1
#         correct += predicted.eq(label).sum().item()

#         progress_bar(i, (50000), 'Acc: %.3f%% (%d/%d)'
#                             % (100.*correct/total, correct, total))

    influence_score_list = dataset_pruning(output_path,classifier,data=[features,labels])
    influence_score_list = np.array(influence_score_list)
    
    
    
    if args.dataset=='cifar10':
        S = (-1/50000) * influence_score_list

    scio.savemat('influence_score.mat', {'data':S})
    dict_ = scio.loadmat('influence_score.mat') # 输出的为dict字典类型
    print(type(dict_['data'])) # numpy.ndarray
    
    # Dataset pruning..
    W = m_guided_opt(S,args.m)
    selected_index = (W==0)
    #Pruned dataset constructing..
    if args.dataset=='cifar10':
        pruned_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
        pruned_dataset.data = pruned_dataset.data[selected_index]
        pruned_dataset.targets = list(np.array(pruned_dataset.targets)[selected_index])
        pruned_loader = torch.utils.data.DataLoader(
        pruned_dataset, batch_size=128, shuffle=True, num_workers=2)
    # print(len(pruned_dataset))

    #Pruned dataset training...
    net = model.ResNet18()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(0, 1):
        net = train(epoch,pruned_loader,net,optimizer)
        # test(epoch,testloader,net,dataset=args.dataset)
        test(epoch,testloader,net)
        scheduler.step()



