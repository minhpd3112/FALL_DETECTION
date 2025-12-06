import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('..')

class Feeder(Dataset):
    """
    Feeder for Fall Detection Dataset
    Data format: (N, T, V, C) = (N, 30, 17, 2)
    """
    def __init__(self, data_path, label_path=None, split='train', 
                 random_choose=False, random_shift=False,
                 random_move=False, window_size=-1, 
                 normalization=False, debug=False,
                 bone=False, vel=False):
        """
        Args:
            data_path: path to X_{split}.npy
            label_path: path to y_{split}.npy
            split: 'train', 'val', or 'test'
            random_choose: randomly choose a portion of frames
            random_shift: randomly shift sequence
            random_move: randomly move sequence
            window_size: sequence length (-1 = use all)
            normalization: normalize data
            debug: use only first 100 samples
            bone: use bone modality
            vel: use velocity modality
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.bone = bone
        self.vel = vel
        
        self.load_data()
        if normalization:
            self.get_mean_map()
    
    def load_data(self):
        """
        Load preprocessed data
        Expected format: (N, T, V, C) where
            N = number of samples
            T = 30 frames
            V = 17 keypoints (COCO format)
            C = 2 (x, y coordinates, already normalized)
        """
        # Load data
        self.data = np.load(self.data_path)  # (N,T,V,C)
        self.label = np.load(self.label_path)  # (N,)
        
        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
        
        N, T, V, C = self.data.shape
        
        # Convert to BlockGCN format: (N, C, T, V, M)
        # M = max persons (always 1 for our dataset)
        # C = channels (2 for x,y)
        # T = temporal frames (30)
        # V = vertices/keypoints (17)
        
        # Add person dimension M=1
        self.data = self.data[..., np.newaxis]  # (N, T, V, C, 1)
        
        # Transpose to (N, C, T, V, M)
        self.data = self.data.transpose(0, 3, 1, 2, 4)
        
        self.sample_name = [f'{self.split}_{i}' for i in range(len(self.data))]
    
    def get_mean_map(self):
        """Calculate mean and std for normalization"""
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    
    def __len__(self):
        return len(self.label)
    
    def __iter__(self):
        return self
    
    def __getitem__(self, index):
        data_numpy = self.data[index]  # (C, T, V, M)
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        
        # For our simple case, we don't need complex augmentations yet
        # Can add later if needed
        
        # Return joint data (for now, same as data_numpy)
        joint = data_numpy
        
        return joint, data_numpy, label, index
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
