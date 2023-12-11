from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
import os 
import torch
from scipy.sparse.linalg import eigs
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN

from skimage.draw import line,line_aa


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
        self.d_pairs_per_map = 20

        # load preprocessed data if exists
        self.pre_process_dir = self.data_dir + 'processed/'
        if os.path.exists(self.pre_process_dir):
            print('Loading preprocessed data')
            data = np.load(self.pre_process_dir + 'preprocessed.npz')
            self.kp = data['kp'] # (N, num_cluster, 2)
            self.maps = data['maps'] # (N, H, W)
            self.labels = data['labels'] # (N * dpair_per_map)
            self.lines = data['lines'] # (N * dpair_per_map, 4)

            # data_len = len(self.maps)
            # start = rank * data_len // world_size
            # end = (rank + 1) * data_len // world_size
            # self.kp = self.kp[start:end]
            # self.maps = self.maps[start:end]
            
        else:
            self._preprocess()

        if normalize:
            # normalize kp to [-1,1]
            image_size = self.maps[0].shape[0]
            self.kp = (self.kp - (image_size)/2) / (image_size/2)
            self.lines = (self.lines - (image_size)/2) / (image_size/2)
            
        print('Loaded {} maps, {} kps'.format(len(self.maps), len(self.kp)))

    def _preprocess(self, cluster_method='kmeans'):
        
        print('Preprocessing data...')

        d_pairs_per_map = self.d_pairs_per_map
        max_intersect = d_pairs_per_map // 2
        lines = []
        labels = []

        for file_name in tqdm(os.listdir(self.data_dir)):
            if file_name.endswith('.npz'):
                data = np.load(self.data_dir + file_name)
                traversible = data['traversible']
                kp = data['kp']

             
                    
                # generate line pairs
                n_intersect = 0
                n = 0
                while True:
                    # randomly sample two points
                    p = np.random.choice(kp.shape[0], size=2, replace=False)
                    # check if the line between the two points intersect with obstacles
                    x1, y1 = kp[p[0]]
                    x2, y2 = kp[p[1]]
                
                    if np.linalg.norm(kp[p[0]] - kp[p[1]]) < 10:
                        continue
                    img = np.zeros_like(traversible)
                    rr, cc,v = line_aa(x1, y1, x2, y2)
                    img[rr, cc] = 1

                    is_intersect = np.any(img[traversible == 0])

                    # need to balance data
                    if is_intersect:
                        if n_intersect < max_intersect:
                            n_intersect += 1
                        else:
                            continue
                        
                    lines.append(np.array([x1, y1, x2, y2]))
                    labels.append(is_intersect)
                    n += 1

                    if n >= d_pairs_per_map:
                        break

                # generate clusters
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
        self.lines = np.stack(lines, axis=0) # (N * dpair_per_map, 2)
        N = self.maps.shape[0]
        self.lines = np.reshape(self.lines, (N,-1,4)) # (N, dpair_per_map, 4)
        self.labels = np.stack(labels, axis=0) # (N, dpair_per_map)
        self.labels = np.reshape(self.labels, (N,-1)) # (N, dpair_per_map)
        
        # save preprocessed data
        os.makedirs(self.pre_process_dir, exist_ok=True)
        np.savez_compressed(self.pre_process_dir + 'preprocessed.npz', 
                            kp=self.kp,
                            lines=self.lines, labels=self.labels, maps=self.maps)
    def __len__(self):
        return self.kp.shape[0]
    
    def __getitem__(self, idx):
        """
        Returns: map, kp, lines, labels
        """
        kp = self.kp[idx] # (num_cluster, 2)
        tra_map = self.maps[idx] # (H, W)
        lines = self.lines[idx] # (dpair_per_map, 4)
        labels = self.labels[idx] # (dpair_per_map)
        
        return torch.from_numpy(tra_map).float().unsqueeze(0), torch.from_numpy(kp).float(), \
            torch.from_numpy(lines).float(), torch.from_numpy(labels).float()
    

class KPLDataset(Dataset):
    
    def __init__(self, split='train', normalize=True, rank=0, world_size=1):
        self.data_dir = DATA_PATH[split]
        self.maps = []
        self.lines = []
        self.labels = []
        self.d_pairs_per_map = 50

        # load preprocessed data if exists
        self.pre_process_dir = self.data_dir + 'kpl_processed/'
        
        if os.path.exists(self.pre_process_dir):
            print('Loading preprocessed data')
            data = np.load(self.pre_process_dir + 'preprocessed.npz')
            self.maps = data['maps']
            self.lines = data['lines']
            self.labels = data['labels']

            # data_len = len(self.lines)
            # start = rank * data_len // world_size
            # end = (rank + 1) * data_len // world_size
            # self.maps = self.maps[start:end]
            # self.lines = self.lines[start:end]
            # self.labels = self.labels[start:end]
            
        else:
            self._preprocess()

        if normalize:
            # normalize kp to [-1,1]
            image_size = self.maps[0].shape[0]
            self.lines = (self.lines - (image_size)/2) / (image_size/2)
            
        print('Loaded {} maps, {} lines'.format(len(self.maps), len(self.lines)))

    def _preprocess(self):
        
        d_pairs_per_map = self.d_pairs_per_map
        max_intersect = d_pairs_per_map // 2
        lines = []
        labels = []
        print('Preprocessing data...')
        for i, file_name in enumerate(tqdm(os.listdir(self.data_dir))):
            if i > 1000:
                break
            if file_name.endswith('.npz'):
                data = np.load(self.data_dir + file_name)
                traversible = data['traversible']
                kp = data['kp']

                n_intersect = 0
                n = 0


                while True:
                    # randomly sample two points
                    p = np.random.choice(kp.shape[0], size=2, replace=False)
                    # check if the line between the two points intersect with obstacles
                    x1, y1 = kp[p[0]]
                    x2, y2 = kp[p[1]]
                
                    if np.linalg.norm(kp[p[0]] - kp[p[1]]) < 10:
                        continue
                    img = np.zeros_like(traversible)
                    rr, cc,v = line_aa(x1, y1, x2, y2)
                    img[rr, cc] = 1

                    is_intersect = np.any(img[traversible == 0])

                    # need to balance data
                    if is_intersect:
                        if n_intersect < max_intersect:
                            n_intersect += 1
                        else:
                            continue
                        
                    lines.append(np.array([x1, y1, x2, y2]))
                    labels.append(is_intersect)
                    n += 1

                    if n >= d_pairs_per_map:
                        break

                self.maps.append(traversible)

        
        self.maps = np.stack(self.maps, axis=0) # (N, H, W)
        self.lines = np.stack(lines, axis=0) # (N* dpair_per_map, 2)
        self.labels = np.stack(labels, axis=0) # (N * dpair_per_map)
        
        # save preprocessed data
        os.makedirs(self.pre_process_dir, exist_ok=True)
        np.savez_compressed(self.pre_process_dir + 'preprocessed.npz', lines=self.lines, labels=self.labels, maps=self.maps)
        
    def __len__(self):
        return self.lines.shape[0]
    
    def __getitem__(self, idx):
        l = self.lines[idx] # [4]
        label = self.labels[idx]
        tra_map = self.maps[idx//self.d_pairs_per_map]

        # # visualize
        return torch.from_numpy(tra_map).float().unsqueeze(0), torch.from_numpy(l).float(), torch.from_numpy(np.array([label])).float()