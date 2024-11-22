from __future__ import print_function
from tqdm import tqdm
import pickle
import argparse
from nuscenes import NuScenes
import numpy as np
import torch

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


def svd():
    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval', help='Root directory of the NuScenes dataset')
    parser.add_argument('--data', type=str, default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_2.pkl", help='Path to detections data')
    parser.add_argument('--output_path', type=str, default="svd_matrices.pkl", help='Path to save the SVD projection matrices')

    args = parser.parse_args()
    data = args.data
    data_root = args.data_root
    version = args.version

    pcds_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}
    fvs_all = {track_name: [] for track_name in NUSCENES_TRACKING_NAMES}

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    with open(data, 'rb') as f:
        all_results = pickle.load(f)

    processed_scene_tokens = set()

    for sample, sample_data in tqdm(all_results.items()):

        scene_token = nusc.get('sample', sample)['scene_token']

        if scene_token in processed_scene_tokens:
            continue

        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        while current_sample_token != '':

            for i, item in enumerate(all_results[current_sample_token]):
                for name in NUSCENES_TRACKING_NAMES:
                    for dets_outputs in item[name]:
                        if dets_outputs['pred_score'] > 0.37:
                            pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                            pcds_all[name].append(pcd_feature)
                            fvs_all[name].append(dets_outputs['feature_vector'])

            current_sample_token = nusc.get('sample', current_sample_token)['next']

        processed_scene_tokens.add(scene_token)

    # Dictionary to store the projection matrices
    svd_matrices = {}
    mean_vectors = {}
    std_vectors = {}

    # Perform SVD for each tracking name
    for name in NUSCENES_TRACKING_NAMES:
        if len(pcds_all[name]) == 0:
            continue

        # Stack and flatten the features (shape: (m, 4608))
        pcds_all[name] = np.vstack(pcds_all[name])
        num_samples = pcds_all[name].shape[0]
        pcds_flat = pcds_all[name].reshape(num_samples, -1)

        # Center the data and normalize them
        mean_vector = np.mean(pcds_flat, axis=0)
        std_vector = np.std(pcds_flat, axis=0)
        std_vector[std_vector == 0] = 1e-8  # Avoid division by zero

        pcds_flat = ( pcds_flat - mean_vector ) / std_vector # mean shift and normalize

        # Convert to tensor
        pcds_flat_tensor = torch.tensor(pcds_flat, dtype=torch.float32)

        # Run SVD
        print(f"Running SVD for {name}...")
        U, S, Vh = torch.linalg.svd(pcds_flat_tensor, full_matrices=False)

        # Save the projection matrix for this tracking name
        svd_matrices[name] = Vh.cpu().numpy()
        mean_vectors[name] = mean_vector
        std_vectors[name] = std_vector

        print(f"Completed SVD for {name}.")

    # Save the SVD matrices
    with open(args.output_path, 'wb') as f:
        pickle.dump({'svd_matrices': svd_matrices, 
                    'mean_vectors': mean_vectors, 
                    'std_vectors': std_vectors}, f)

    print(f"Saved SVD projection matrices to {args.output_path}.")

if __name__ == "__main__":
    svd()