from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import skfmm
import torch

def get_dist_map(traversible: np.ndarray, 
                 goal_map: np.ndarray = None, 
                 goal_pose: np.ndarray = None) -> np.ndarray:
    """
    Compute the distance map to the goal
    args:
        traversible: binary map of traversible area, 1 means traversible
        goal_map: binary map of goal, 1 means goal
        goal_pose: goal pose, 
    """
    if goal_map is None and goal_pose is None:
        raise ValueError("Either goal_map or goal_pose must be provided")
    
    traversible_ma = np.ma.masked_values(traversible * 1, 0)
    if goal_pose is not None:
        traversible_ma[goal_pose[0], goal_pose[1]] = 0
    else:
        traversible_ma[goal_map == 1] = 0

    dist_map = skfmm.distance(traversible_ma, dx=1)
    dist_map = dist_map.filled(dist_map.max())

    return dist_map

    
DATA_PATH = {
    'train': 'data/maze_192/train/',
    'test': 'data/maze_192/test/',
}
class I2PDataset(Dataset):
    
    def __init__(self, split='train'):
        self.data_dir = DATA_PATH[split]
        self.maps = []
        self.goals = []
        self.data = [] # (map_id, goal)

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.npz'):
                data = np.load(self.data_dir + file_name)
                traversible = data['traversible']
                goals = data['goals']
                for g in goals:
                    self.data.append((len(self.maps), g))
                self.maps.append(traversible)

        # sample goals
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        tra_map = self.maps[data[0]].astype(np.float32)

        tra_map[tra_map == 0] = -5 # obstacle
        tra_map[data[1][0], data[1][1]] = -10 # goal
        tra_map[tra_map == 1] = 0 # traversible
        goal_xy = data[1] / 192

        dist_map = get_dist_map(self.maps[data[0]], goal_pose=data[1])
        dist_map = dist_map.astype(np.float32) / 500

        return torch.from_numpy(tra_map).unsqueeze(0), torch.from_numpy(dist_map).unsqueeze(0), torch.from_numpy(goal_xy) # (2,H,W), (1,H,W)
    