import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# COCO 17 keypoints
num_node = 17
self_link = [(i, i) for i in range(num_node)]

# COCO skeleton connections (0-indexed)
# 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear
# 5:L_Shoulder, 6:R_Shoulder, 7:L_Elbow, 8:R_Elbow
# 9:L_Wrist, 10:R_Wrist, 11:L_Hip, 12:R_Hip
# 13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle

inward_ori_index = [
    # Head connections
    (1, 0),   # L_Eye -> Nose
    (2, 0),   # R_Eye -> Nose  
    (3, 1),   # L_Ear -> L_Eye
    (4, 2),   # R_Ear -> R_Eye
    (5, 0),   # L_Shoulder -> Nose (through neck, simplified)
    (6, 0),   # R_Shoulder -> Nose (through neck, simplified)
    
    # Left arm
    (7, 5),   # L_Elbow -> L_Shoulder
    (9, 7),   # L_Wrist -> L_Elbow
    
    # Right arm  
    (8, 6),   # R_Elbow -> R_Shoulder
    (10, 8),  # R_Wrist -> R_Elbow
    
    # Torso
    (11, 5),  # L_Hip -> L_Shoulder
    (12, 6),  # R_Hip -> R_Shoulder
    
    # Left leg
    (13, 11), # L_Knee -> L_Hip
    (15, 13), # L_Ankle -> L_Knee
    
    # Right leg
    (14, 12), # R_Knee -> R_Hip
    (16, 14), # R_Ankle -> R_Knee
]

inward = inward_ori_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
