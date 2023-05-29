import os
import torch
import torchvision
import argparse
import sys
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from dataset import iclevrDataset
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.evaluator import evaluation_model

os.chdir(sys.path[0])

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=60, help='Number of training epochs')
    parser.add_argument('--batch_size', default=28, help='Size of batches')
    parser.add_argument('--lr', default=0.0002, help='Learning rate')
    parser.add_argument('--c_dim', default=4, help='Condition dimension')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--model_path', default='ckpt', help='Path to save model checkpoint')
    parser.add_argument('--log', default='log/origin', help='Path to save log')
    parser.add_argument('--timesteps', default=1000, help='Time step for diffusion')
    parser.add_argument('--test_file', default='test.json', help='Test file')
    parser.add_argument('--test_batch_size', default=32, help='Test batch size')
    parser.add_argument('--figure_file', default='figure/origin', help='Figure file')
    parser.add_argument('--resume', default=False, help='Continue for training')
    parser.add_argument('--ckpt', default='net.pth', help='Checkpoint for network')
    
    return parser.parse_args()

class ConditionedUNet(nn.Module):
    def __init__(self, args, num_class = 24, embed_size = 512):
        super().__init__() 
        
        self.model = UNet2DModel(
            sample_size = 64,
            in_channels =  3,
            out_channels = 3,
            layers_per_block = 2,
            class_embed_type = None,
            block_out_channels = (128, 256, 256, 512, 512, 1024),
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",    
                "DownBlock2D",    
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D", 
            ), 
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",         
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        self.model.class_embedding = nn.Linear(num_class, embed_size)
    
    def forward(self, x, t, y):
        return self.model(x, t, class_labels = y).sample 

def train(net, args, train_loader, test_loader, device, noise_scheduler):
    net.to(device)
    
    # Loss functions
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    
    writer = SummaryWriter(args.log)
    
    iters = 0 
    best_acc = 0
    
    for epoch in tqdm(range(args.epochs)):
        net.train()
        total_loss = 0
        for idx, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            
            noise = torch.randn_like(img)
            timesteps = torch.randint(0, args.timesteps - 1, (img.shape[0],)).long().to(device)
            noisy_img = noise_scheduler.add_noise(img, noise, timesteps)
            
            # Get prediction
            pred = net(noisy_img, timesteps, label)
            
            # Calculate loss
            loss = criterion(pred, noise) # How close is the output to the noise

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if iters % 1000 == 0:
                writer.add_scalar('Train Step loss', loss.item(), iters)
                wandb.log({"Training Step loss": loss.item()})
            
            iters += 1

           
        writer.add_scalar('Train Epoch Loss', total_loss / len(train_loader), epoch)
        print(f'Train Epoch_{epoch} loss: {total_loss / len(train_loader) :.5f}\n')
        
        acc = test(net, args, test_loader, device, noise_scheduler, epoch)
        writer.add_scalar('Testing acc', acc, epoch)
        print(f'Testing acc: {acc:.2f}\n')
        
        eval_metric = {"Training Epoch Loss": total_loss / len(train_loader),
                       "Testing acc": acc
        }
        wandb.log(eval_metric)
        
        if acc > best_acc:
            best_acc = acc 
            torch.save(net.state_dict(), os.path.join(args.model_path, args.ckpt))
        
          
def test(net, args, test_loader, device, noise_scheduler, epoch = None):
    net.to(device)
    net.eval()
    
    score = evaluation_model()
    acc = 0
    
    if args.test_file == 'test.json':
        filename = 'test'
    else:
        filename = 'new_test'
    
    with torch.no_grad():
        for label in test_loader:
            label = label.to(device)
            noise = torch.randn(label.shape[0], 3, 64, 64).to(device)
            
            for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
                # Get model predict
                residual = net(noise, t, label)
                
                # Update sample with step
                noise = noise_scheduler.step(residual, t, noise).prev_sample
            
            img = noise
            
        acc = score.eval(img, label)
        if epoch != None:
            save_image(make_grid(img, nrow = 8, normalize = True), os.path.join(args.figure_file, f'{filename}_{epoch}_{acc:.2f}.png'))
        else:
            save_image(make_grid(img, nrow = 8, normalize = True), os.path.join(args.figure_file, f'{filename}_{acc:.2f}_final.png'))
    
    return acc
            

if __name__ == '__main__':
    args = parse_arg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader = DataLoader(iclevrDataset(mode = 'train'), batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(iclevrDataset(mode = 'test', test_file = args.test_file), batch_size = args.test_batch_size)
    
    net = ConditionedUNet(args)
    noise_scheduler = DDPMScheduler(args.timesteps)
    
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(args.figure_file, exist_ok=True)
    
    if args.test_only:
        net.load_state_dict(torch.load(os.path.join(args.model_path, args.ckpt)))
        acc = test(net, args, test_loader, device, noise_scheduler)
        print(f'acc: {acc}')
        
    else:
        wandb.init(
            project = 'Deep Learning Lab7',
            config = {"batch_size":28, 
                    "epoch": 60, 
                    "embedding": "nn.Linear", 
                    "Type": "DDDDAA",
                    "Block_size": 'bigger',
                    "Resume": False
            },
            name = "Big block"
        )
        
        if args.resume:
            net.load_state_dict(torch.load(os.path.join(args.model_path, args.ckpt)))
        
        train(net, args, train_loader, test_loader, device, noise_scheduler)
        wandb.finish()