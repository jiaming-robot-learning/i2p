
"""
Generate pairs of image for training and testing
"""
import subprocess
from skimage import io
import skimage
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import random
# import skfmm
import os

ROOT_DIR = '/home/users/jiaming/i2p/'

# def get_dist_map(traversible: np.ndarray, 
#                  goal_map: np.ndarray = None, 
#                  goal_pose: np.ndarray = None) -> np.ndarray:
#     """
#     Compute the distance map to the goal
#     args:
#         traversible: binary map of traversible area, 1 means traversible
#         goal_map: binary map of goal, 1 means goal
#         goal_pose: goal pose, in grid coordinates (x,y)
#     """
#     if goal_map is None and goal_pose is None:
#         raise ValueError("Either goal_map or goal_pose must be provided")
    
#     traversible_ma = np.ma.masked_values(traversible * 1, 0)
#     if goal_pose is not None:
#         traversible_ma[goal_pose[1], goal_pose[0]] = 0
#     else:
#         traversible_ma[goal_map == 1] = 0

#     dist_map = skfmm.distance(traversible_ma, dx=1)
#     dist_map = dist_map.filled(dist_map.max())

#     return dist_map


# def sample_start_goal(current_map):
#     max_sampling_iter = 1000
#     dilate_size = 7
#     goal_min_distance = 50
    
#     traversible = current_map.copy()
#     traversible = skimage.morphology.binary_erosion(traversible, np.ones((dilate_size,dilate_size)))
#     nav_idx = np.nonzero(traversible == 1)
#     num_idx = nav_idx[0].shape[0]
#     success = False

#     for i in range(max_sampling_iter):
#         start_idx = random.randint(0, num_idx - 1)
#         goal_idx = random.randint(0, num_idx - 1)
#         start = np.array([nav_idx[0][start_idx], nav_idx[1][start_idx]])[[1,0]] # follow the convention of [y,x]
#         goal = np.array([nav_idx[0][goal_idx], nav_idx[1][goal_idx]])[[1,0]] # follow the convention of [y,x]

#         # make sure the start and goal are far enough, and the goal is reachable
#         dist_map = get_dist_map(traversible, goal_pose=goal)
#         max_dist = np.max(dist_map)
#         start_dist = dist_map[start[1], start[0]]
#         if start_dist > goal_min_distance and start_dist < max_dist:
#             success = True
#             break
#     if not success:
#         # raise ValueError(f"Failed to sample a start and goal for scene {self._current_scene}")
#         return None, None
    
#     return start, goal

def gen_raw_maze(file_name):
    js_dir = ROOT_DIR + 'i2p/dataset/js/'
    subprocess.Popen(["node", 'maze.js'], cwd=js_dir).wait()
    goal_per_map = 20
    
    # read the generate maze
    im = io.imread(js_dir + 'maze.png')
    im = rgb2gray(im[:,:,:3])[9:-9, 9:-9]
    traversible = im.astype(bool)
    traversible = skimage.morphology.binary_dilation(traversible, np.ones((6,6)))

    traversible_idx = np.nonzero(traversible == 1)
    # randomly sample N goals
    goal_ids = np.random.randint(0, traversible_idx[0].shape[0], size=goal_per_map)
    goals = np.array([traversible_idx[0][goal_ids], traversible_idx[1][goal_ids]]).T

    np.savez_compressed(file_name, traversible=traversible, goals=goals)

    print(f'Generated {file_name}')

    
if __name__ == '__main__':

    data_dir = ROOT_DIR + 'i2p/data/maze_192/train/'
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(20000):
        file_name = data_dir + 'maze_' + str(i) + '.npz'
        gen_raw_maze(file_name)

    data_dir = ROOT_DIR + 'i2p/data/maze_192/test/'
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(1000):
        file_name = data_dir + 'maze_' + str(i) + '.npz'
        gen_raw_maze(file_name)