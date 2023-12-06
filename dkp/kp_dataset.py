from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
import os 
import torch
from scipy.sparse.linalg import eigs
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
DATA_PATH = {
    'train': 'data/maze_192_kp/train/',
    'test': 'data/maze_192_kp/test/',
}

def visualize_kp(tra_map, kps, normalized=True, ax=None):
    m = tra_map.copy().astype(np.float32)
    m[~tra_map] = 255 # obstacle
    m[tra_map] = 0 # traversible

    # unnormalize kp
    if normalized:
        kps = kps  * (tra_map.shape[0]/2) + (tra_map.shape[0]/2)
    kps = kps.astype(np.int32)
    for kp in kps:
        m[kp[0], kp[1]] = 128

    if ax is None:
        plt.imshow(m)
        plt.show()
    else:
        ax.imshow(m)
        ax.axis('off')
        

class KPDataset(Dataset):
    
    def __init__(self, split='train', normalize=True, rank=0, world_size=1):
        self.data_dir = DATA_PATH[split]
        self.maps = []
        self.kp_per_map = 0
        self.kp = []

        # load preprocessed data if exists
        pre_process_dir = self.data_dir + 'preprocess/'
        if os.path.exists(pre_process_dir):
            print('Loading preprocessed data')
            data = np.load(pre_process_dir + 'preprocessed.npz')
            self.kp = data['kp']
            self.maps = data['maps']
            data_len = len(self.kp)
            start = rank * data_len // world_size
            end = (rank + 1) * data_len // world_size
            self.kp = self.kp[start:end]
            self.maps = self.maps[start:end]
            
        else:
            self._preprocess()

        if normalize:
            # normalize kp to [-1,1]
            image_size = self.maps[0].shape[0]
            self.kp = (self.kp - (image_size)/2) / (image_size/2)
            
        print('Loaded {} maps, {} keypoints'.format(len(self.maps), len(self.kp)))

    def _preprocess(self, cluster_method='kmeans'):
        
        print('Preprocessing data...')
        for file_name in tqdm(os.listdir(self.data_dir)):
            if file_name.endswith('.npz'):
                data = np.load(self.data_dir + file_name)
                traversible = data['traversible']
                kp = data['kp']

                # visualize
                # fig, ax = plt.subplots(1,3, figsize=(10,5))
                # visualize_kp(traversible, kp, normalized=False, ax=ax[0])
                
                # kmeans = KMeans(n_clusters=50, random_state=0).fit(kp)
                # kp_c = kmeans.cluster_centers_
                # visualize_kp(traversible, kp_c, normalized=False, ax=ax[1])

                # db = DBSCAN(eps=0.1, min_samples=20).fit(kp)
                # kp_c = kp[db.labels_ != -1]
                # visualize_kp(traversible, kp_c, normalized=False, ax=ax[2])

                # fig.show()

                if cluster_method == 'kmeans':
                    # print('Preprocessing with kmeans for {}...'.format(file_name))
                    n_clusters = 50
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(kp)
                    kp = kmeans.cluster_centers_
                    self.kp_per_map = n_clusters

                elif cluster_method == 'dbscan':
                    db = DBSCAN(eps=0.1, min_samples=100).fit(kp)
                    kp = kp[db.labels_ != -1]
                elif cluster_method == None:
                    pass
                else:
                    raise NotImplementedError
                    
                self.kp.append(kp) # (N, 2)
                self.maps.append(traversible)

        
        self.maps = np.stack(self.maps, axis=0) # (N, H, W)
        self.kp = np.stack(self.kp, axis=0) # (N, num_cluster, 2)
        
        # save preprocessed data
        pre_process_dir = self.data_dir + 'preprocess/'
        os.makedirs(pre_process_dir, exist_ok=True)
        np.savez_compressed(pre_process_dir + 'preprocessed.npz', kp=self.kp, maps=self.maps)
        
    def __len__(self):
        return self.kp.shape[0]
    
    def __getitem__(self, idx):
        kp = self.kp[idx] # (num_cluster, 2)
        tra_map = self.maps[idx] # (H, W)

        # # visualize
        return torch.from_numpy(tra_map).float().unsqueeze(0), torch.from_numpy(kp).float()
    