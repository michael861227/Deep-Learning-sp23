import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):

        self.seed_is_set = False
        self.data_path = args.data_root
        self.mode = mode 
        self.seq_length = args.n_past + args.n_future

        min, max = [0.42638585, -0.3080245 ,  0.19146784], [0.42850533, 0.54029283, 0.12564658]

        self.videos = []
        self.positions = []
        self.actions = []
        videos = os.listdir(self.data_path + mode + '/')
        for vid in videos:
            subvideos = os.listdir(self.data_path + mode + '/' + vid + '/')
            for svid in subvideos:
                self.videos.append(mode + '/' + vid + '/' + svid + '/')
                pos = np.loadtxt(self.data_path + mode + '/' + vid + '/' + svid + '/endeffector_positions.csv', delimiter=',')
                self.positions.append((pos-min)/max)
                action = np.loadtxt(self.data_path + mode + '/' + vid + '/' + svid + '/actions.csv', delimiter=',')
                self.actions.append(action)

        self.length = len(self.videos)
        self.positions = np.asarray(self.positions)
        self.actions = np.asarray(self.actions)
        self.transform = transform
            
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return self.length
    
    def load_img(self, video, frame):
        img = cv2.imread(self.data_path + video + f'/{frame}.png')
        return self.transform((cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0)
        
    def get_seq(self, index):
        video = self.videos[index]

        seq = torch.stack([self.load_img(video, i).to(torch.float32) for i in range(self.seq_length)], dim=0)
        
        return seq

    
    def get_csv(self, index):
        # start = 0
        position = self.positions[index, :self.seq_length]
        action = self.actions[index, :self.seq_length]

        condition = np.hstack((position, action))
        
        return torch.from_numpy(condition).to(torch.float32)

    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond =  self.get_csv(index)

        return seq, cond
