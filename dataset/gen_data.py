
import subprocess
from skimage import io
import skimage
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import os 
from scipy.sparse.linalg import eigs
import tqdm

ROOT_DIR = '/home/users/jiaming/i2p/'

K = 60 # number of eigenvectors
TN = 80 # top TN nodes for each eigenvector to sample from 
N=10 # number of nodes to draw for each eigenvector
T = 0.001 # temperature for softmax

def map_to_graph(a):
    """
    Convert a binary map to a graph
    """
    # define grid graph according to the shape of a
    G = nx.grid_2d_graph(*a.shape)

    # remove those nodes where the corresponding value is != 0
    for val,node in zip(a.ravel(), sorted(G.nodes())):
        if not val:
            G.remove_node(node) # not traversible

    return G

def draw_graph(G, attr,ax = None):
    """
    Draw the graph
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # coordinate rotation
    pos = {(x,y):(y,-x) for x,y in G.nodes()}
    vals = nx.get_node_attributes(G, attr)
    vals = [vals[i] for i in G.nodes()]

    nx.draw(G, pos=pos, 
            ax=ax,
            node_color=vals,
            width = 2,
            node_size=8,
            vmin=min(vals),
            vmax=max(vals),
            )


def sample_kp(tra_map, n_samples=10000):
    """
    For each map
    1. compute the graph laplacian
    2. compute the eigenvectors of the graph laplacian
    3. sample from the graph according to the probability defined by each eigenvectors, using softmax
    """
    # convert map to graph, and compute eigenvectors of the graph laplacian
    graph = map_to_graph(tra_map)
    lap_matrix = nx.laplacian_matrix(graph).asfptype()
    vals, vecs = eigs(lap_matrix, k=K, which='LR')
    vecs = vecs.real

    # compute softmax for top K eigenvectors
    sm_vecs = np.zeros_like(vecs)
    for i in range(K):
        sm = np.exp(vecs[:, i]/T)
        sm = sm / np.sum(sm, axis=0)
        sm_vecs[:, i] = sm

    # ----------------
    # sample from the graph according to the softmax prob
    # ----------------
    node_count = np.zeros_like(sm_vecs[:, 0])
    sampled_coords = []
    coords = np.array(list(graph.nodes()))
    
    for _ in range(n_samples):
        # first select which eigenvector to sample
        eig = np.random.choice(K, p=np.ones(K)/K)
        
        # then sample from the eigenvector
        sm = sm_vecs[:, eig]
        node = np.random.choice(len(graph.nodes()), p=sm)
        node_count[node] += 1

        sampled_coords.append(coords[node])
        
    sampled_coords = np.stack(sampled_coords, axis=0)
    
    # visualize
    visualize = False
    if visualize:
        nx.set_node_attributes(graph, dict(zip(graph.nodes(), node_count)), 'sample')
        draw_graph(graph,'sample')
        plt.show()

        m = tra_map.copy().astype(np.float32)
        m[~tra_map] = 100 # obstacle
        m[tra_map] = 0 # traversible
        for c in sampled_coords:
            m[c[0], c[1]] += 1
        plt.imshow(m)

    return sampled_coords

def gen_single_maze(file_name, goal_per_map=0, kp_per_map=0):
    js_dir = ROOT_DIR + 'i2p/dataset/js/'
    subprocess.Popen(["node", 'maze.js'], cwd=js_dir).wait()
    
    # read the generate maze
    im = io.imread(js_dir + 'maze.png')
    im = rgb2gray(im[:,:,:3])[9:-9, 9:-9]
    traversible = im.astype(bool)
    traversible = skimage.morphology.binary_dilation(traversible, np.ones((6,6)))

    data = {
        'traversible': traversible,
    }
    # if sample goals
    if goal_per_map > 0:
        traversible_idx = np.nonzero(traversible == 1)
        goal_ids = np.random.randint(0, traversible_idx[0].shape[0], size=goal_per_map)
        goals = np.array([traversible_idx[0][goal_ids], traversible_idx[1][goal_ids]]).T
        data['goals'] = goals
    
    # if sample kps
    if kp_per_map > 0:
        data['kp'] = sample_kp(traversible, n_samples=kp_per_map)
        
    np.savez_compressed(file_name, **data)

    print(f'Generated {file_name}')

    
if __name__ == '__main__':

    for split in ['train']:
        data_dir = ROOT_DIR + 'i2p/data/maze_192_kp/' + split + '/'
        os.makedirs(data_dir, exist_ok=True)
        
        for i in range(14819, 50000):
            file_name = data_dir + 'maze_' + str(i) + '.npz'
            gen_single_maze(file_name, kp_per_map=10000)
            