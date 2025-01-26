from pathlib import Path
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm


ROOT = '/scratch/Peract_packaged/'
STORE_PATH = '/data/user_data/ngkanats/Peract_zarr/'
READ_EVERY = 100  # in episodes
STORE_EVERY = 1  # in keyposes
NCAM = 4
IM_SIZE = 256
def main():
    ###############################
    # save data
    ###############################
    # create zarr file
    save_dir = "data_test"


    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta

    # action = np.zeros( (14,) )
    # pointcloud = np.zeros( (10000,6) )
    # state = np.zeros( (14,) )
    # end_idx = [13]


        
    state_arrays = np.zeros( (100,14) )
    cloud_arrays = np.zeros( (100,10000,6) )
  
        
    action_arrays = np.zeros( (100,14) )
    episode_ends_arrays = np.array([50,100])

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    single_size = 500
    state_chunk_size = (single_size, state_arrays.shape[1])
    point_cloud_chunk_size = (single_size, cloud_arrays.shape[1], cloud_arrays.shape[2])
    action_chunk_size = (single_size, action_arrays.shape[1])
        
    
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
      
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
if __name__ == "__main__":
    main()

    # [frame_ids],  # we use chunk and max_episode_length to index it [0,1,2...n]
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (2, 8), predited action (keypose, next goal position)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (2, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 2, 8), (for traj, first frame is the current state)
    # List of tensors
