import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import copy
import os
import sys
from ResNet import ResNet, BasicBlock, BottleneckBlock
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

os.chdir(sys.path[0])

def parse_arguments():
    parser = ArgumentParser(description = 'ResNet')
    parser.add_argument('-sp', '--save_place', default = './models')
    parser.add_argument('-rp', '--result_place', default = './results')
    parser.add_argument('-b', '--batch_size', default = 8, help = 'Batch size')
    parser.add_argument('-e', '--epochs', default = 10, help = 'Number of epoch')
    parser.add_argument('-lr', '--learning_rate', default = 1e-3, help = 'Learning rate')
    parser.add_argument('-o', '--optimizer', default = torch.optim.SGD, help = 'Optimizer')
    parser.add_argument('-w', '--weight_decay', default = 5e-4, help = 'Weight decay')
    parser.add_argument('-m', '--momentum', default=0.9, help='Momentum factor for SGD')
    parser.add_argument('-l', '--load_model', default = './', help = 'Load weight for the model')
    parser.add_argument('-d', '--dropout', default = 0.25, help = 'Dropout')
    parser.add_argument('-t', '--test', default = False, help = 'Test mode for demo')
    
    return parser.parse_args()


def ResNet18(args, pretrained = False):
    if pretrained:
        model = models.resnet18(pretrained = True)
        model.fc = nn.Sequential(
            nn.Linear(in_features = 512 * 1, out_features = 100),
            nn.ReLU(inplace = True),
            nn.Dropout(p = args.dropout),
            nn.Linear(in_features = 100, out_features = 5)
        )

    else:
        model = ResNet(BasicBlock, layers = [2, 2, 2, 2], dropout = args.dropout, num_classes = 5)
    
    return model

def ResNet50(args, pretrained = False):
    if pretrained:
        model = models.resnet50(pretrained = True)
        model.fc = nn.Sequential(
            nn.Linear(in_features = 512 * 4, out_features = 100),
            nn.ReLU(inplace = True),
            nn.Dropout(p = args.dropout),
            nn.Linear(in_features = 100, out_features = 5)
        )
    
    else:
        model = ResNet(BottleneckBlock, layers = [3, 4, 6, 3], dropout = args.dropout, num_classes = 5)
    
    return model

def train(model_name, model_dict, train_loader, test_loader, ground_truth, args, df, device):
    loss_function = nn.CrossEntropyLoss()
    best_testing_acc = {'without_pretraining': 0, 'with_pretrianing': 0}
    best_weights = {'without_pretraining': None, 'with_pretrianing': None}
    
            
    for model_state, model in model_dict.items():
        model.to(device)
        model.train()
         
        optimizer = args.optimizer(model.parameters(), lr = args.learning_rate, momentum = args.momentum,
                                   weight_decay = args.weight_decay)
        
        acc_test = []
        acc_train = []
        
        print(f'{model_name}_{model_state}')
        for epoch in tqdm(range(1, args.epochs + 1)):
            total_loss = 0
            train_correct = 0
            test_correct = 0
            
            # train
            for img, label in train_loader:
                img, label = img.to(device), label.to(device, dtype = torch.long)
                out = model(img)

                loss = loss_function(out, label)
                total_loss += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                predict = out.max(dim = 1)[1]
                train_correct += (predict == label).sum().item()
            
            total_loss /= len(train_dataset) 
            train_correct = train_correct / len(train_loader.dataset) * 100
            
            print(f'epoch:{epoch:<4d}   loss:{total_loss:.4f}   train_acc:{train_correct:.2f}%')
            
            acc_train.append(train_correct)
            
            
            # test
            model.eval()
            with torch.no_grad():
                for img, label in test_loader:
                    img, label = img.to(device), label.to(device, dtype = torch.long)
                    out = model(img)
                    out = out.max(dim = 1)[1]
                    
                    test_correct += (out == label).sum().item()
                
                test_correct = test_correct * 100 / len(test_loader.dataset)
                acc_test.append(test_correct)
            
            if test_correct > best_testing_acc[model_state]:
                best_testing_acc[model_state] = test_correct
                best_weights[model_state] = copy.deepcopy(model.state_dict())

            
        df[f'Train({model_state})'] = acc_train
        df[f'Test({model_state})'] = acc_test
        
        
        if not os.path.exists(args.save_place):
            os.mkdir(args.save_place)
            
        torch.save(best_weights[model_state], os.path.join(args.save_place, f'{model_name}_{model_state}.pt'))
        
        plot_confusion_matrix(model_name, model, model_state, args, ground_truth, test_loader, device)
        
        with open('demo.txt', 'a') as f:
            f.write(f'{model_name}_{model_state}: {best_testing_acc[model_state]:.2f}%' + '\n')
        
        torch.cuda.empty_cache()

def plot_confusion_matrix(model_name, model, model_state, args, ground_truth, test_loader, device):
    net = model
    net.load_state_dict(torch.load(os.path.join(args.save_place, f'{model_name}_{model_state}.pt')))
    net.to(device)
    
    net.eval()
    with torch.no_grad():
        predict_label = np.array([], dtype = int)
        for img, label in test_loader:
            img, label = img.to(device), label.to(device, dtype = torch.long)
            out = net(img)
            out = out.max(dim = 1)[1]
            
            # use for confusion matrix
            predict_label = np.concatenate((predict_label, out.cpu().numpy()))
    
    cm = confusion_matrix(y_true = ground_truth, y_pred = predict_label, normalize = 'true')
    ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2, 3, 4]).plot(cmap = plt.cm.Blues)
    plt.title(f'{model_name}_{model_state}')
    
    if not os.path.exists(args.result_place):
        os.mkdir(args.result_place)
        
    plt.savefig(os.path.join(args.result_place, f'{model_name}_{model_state}_confusionMatrix.png'))
    plt.close()
    
def Plot(df, model_name):
    flg = plt.figure(figsize = (10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name}_Result')
    for column in df.columns:
        plt.plot(df[column], label = column)
    
    plt.legend(loc = 'lower right')
    
    if not os.path.exists(args.result_place):
        os.mkdir(args.result_place)
    flg.savefig(os.path.join(args.result_place, f'{model_name}_result.png'))
    plt.close() 
    
def test(model_name, model_dict, test_loader, args, device):
    for model_state, model in model_dict.items():
        net = model
        net.load_state_dict(torch.load(os.path.join(args.save_place, f'{model_name}_{model_state}.pt')))
        net.to(device)
        
        net.eval()
        with torch.no_grad():
            test_correct = 0
            
            for img, label in test_loader:
                img, label = img.to(device), label.to(device, dtype = torch.long)
                out = net(img)
                out = out.max(dim = 1)[1]
                
                test_correct += (out == label).sum().item()
            
            test_correct = test_correct / len(test_loader.dataset) * 100
            
            print(f'{model_name}_{model_state}: {test_correct:.2f}%')
            
            with open('demo.txt', 'a') as f:
                f.write(f'{model_name}_{model_state}: {test_correct:.2f}%' + '\n')
                
                
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()
    df = pd.DataFrame()
    networks = {'ResNet50': {'without_pretraining': ResNet50(args, pretrained = False), 'with_pretrianing': ResNet50(args, pretrained = True)},
                'ResNet18': {'without_pretraining': ResNet18(args, pretrained = False), 'with_pretrianing': ResNet18(args, pretrained = True)}    
    }
    
    train_dataset = RetinopathyLoader('./new_train', 'train')
    test_dataset = RetinopathyLoader('./new_test', 'test')
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 4)
    
    if not args.test:
        # use for confusion matrix
        ground_truth = np.array([], dtype = int)
        for _, label in test_loader:
            ground_truth = np.concatenate((ground_truth, label.long().view(-1).numpy()))
        
        for model_name, model_dict in networks.items():
            train(model_name, model_dict, train_loader, test_loader, ground_truth, args, df, device)
            Plot(df, model_name)
    
    else:
        # for model_name, model_dict in networks.items():
            # test(model_name, model_dict, test_loader, args, device)
        with open('demo.txt', 'r') as f:
            for line in f.readlines():
                print(line.strip())



