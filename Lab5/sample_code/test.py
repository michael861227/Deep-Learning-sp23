import argparse
import itertools
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred, plot_test
from train_fixed_prior import kl_annealing

torch.backends.cudnn.benchmark = True
os.chdir(sys.path[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs', help='base directory to save logs')
    parser.add_argument('--model_dir', default='./logs/decay_epoch=0-kl_weight_type=cyclical-tfr_truncate_epoch=10', help='base directory to save logs')
    parser.add_argument('--data_root', default='../', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.02, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--kl_weight_type', default='monotonic', help = 'type of monotonic or cyclical')
    parser.add_argument('--kl_monotonic_truncate_epoch', default = 100)
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--c_dim', type=int, default=7, help='dimensionality of condition')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    saved_model = torch.load(f'{args.model_dir}/model.pth')
    args = saved_model['args']
    
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # ------------ build the models  --------------
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    validate_data = bair_robot_pushing_dataset(args, 'test')
    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)
    
    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    
    # --------- testing loop ------------------------------------
    for epoch in range(0, 1):
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()


        psnr_list = []
        progress = tqdm(total = len(validate_data) // args.batch_size)
        for _ in range(len(validate_data) // args.batch_size):
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            validate_seq = torch.transpose(validate_seq, 0, 1).to(device, dtype=torch.float32)
            validate_cond = torch.transpose(validate_cond, 0, 1).to(device, dtype=torch.float32)

            pred_seq = pred(validate_seq, validate_cond, modules, args, device)
            _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
            psnr_list.append(psnr)
            plot_test(validate_seq, validate_cond, modules, epoch, args, device)
            
            progress.update(1)
        ave_psnr = np.mean(np.concatenate(psnr_list))

        with open('./{}/test_record.txt'.format(args.log_dir), 'w') as test_record:
            test_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

        plot(args)

def plot(args):
    kl_anneal = kl_annealing(args)
    kl = []
    tfr = []
    tfr_ = 1
    
    for epoch in range(0, args.niter):
        kl.append(kl_anneal.get_beta(args = args, epoch = epoch, start_epoch = 0))
        
        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            ## TODO
            tfr_decay_step = 1.0 / (args.tfr_truncate_epoch - args.tfr_start_decay_epoch)
            tfr_ -= tfr_decay_step 
            tfr_ = max(tfr_, 0)
            tfr.append(tfr_)
     
    loss = []
    mse = []
    kld = []
    psnr = []       
    with open(args.log_dir + '/train_record.txt', 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            tmp = line.split(' ')
            if(tmp[0] == "[epoch:") :
                loss.append(float(tmp[3]))
                mse.append(float(tmp[7]))
                kld.append(float(tmp[-1]))
                
            elif(tmp[0] == "======================"):
                psnr.append(float(tmp[4]))

    
    epochs = np.arange(0, 300)
    psnr_epochs = np.arange(0, 300, 5)

    fig, ax1 = plt.subplots()
    plt.title('Training loss/ratio curve')
    plt.xlabel('epochs')
    
    ax2 = ax1.twinx()
    ax1.set_ylabel('loss/psnr')
    ax1.plot(epochs, kld, 'b', label='kld')
    ax1.plot(epochs, loss, 'ro', label='total loss')
    ax1.plot(epochs, mse,'r', label='mse')
    ax1.plot(psnr_epochs, psnr, 'g.', label='psnr')
    ax1.legend()
    ax1.set_ylim([0.0, 30.0])
    
    ax2.set_ylabel('ratio')
    ax2.plot(epochs, tfr, 'm--', label='Teacher ratio')
    ax2.plot(epochs, kl, '--', color='orange', label='KL weight')
    fig.tight_layout()
    ax2.legend()

    plt.savefig(os.path.join(args.log_dir, 'result.png'))
    
if __name__ == '__main__':
    main()
    
        