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

def create_box_annotations(sample_token,nusc):
    ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

    sample = nusc.get('sample', sample_token)
    ann_token = sample['anns']

    for ann in ann_token:

        ann_meta = nusc.get('sample_annotation', ann)
        t_name = ann_meta['category_name']

        for tracking_name in NUSCENES_TRACKING_NAMES:
            if tracking_name in t_name:

                trs = np.array(ann_meta['translation'])

                ground_truths[tracking_name].append(trs)

    return ground_truths


def svd():
    parser = argparse.ArgumentParser(description="TrainVal G2 with lidar and camera detected characteristics")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval', help='Root directory of the NuScenes dataset')
    parser.add_argument('--data', type=str, default="/home/ktsiakas/thesis_new/2D_FEATURE_EXTRACTOR/mrcnn_val_2.pkl", help='Path to detections data')
    parser.add_argument('--output_path', type=str, default="svd_matrices_spatial.pkl", help='Path to save the SVD projection matrices')

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

        current_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

        while current_sample_token != '':

            current_ground_truths = create_box_annotations(current_sample_token, nusc)

            for i, item in enumerate(all_results[current_sample_token]):
                for name in NUSCENES_TRACKING_NAMES:
                    for dets_outputs in item[name]:
                        det_coords = np.array(dets_outputs['box'][:2])

                        for gt_coords in current_ground_truths[name]:
                            gt_coords = np.array(gt_coords[:2])
                            distance = np.linalg.norm(det_coords - gt_coords)

                            if distance < 2:
                                pcd_feature = np.expand_dims(dets_outputs['point_cloud_features'], axis=0)
                                pcds_all[name].append(pcd_feature)
                                fvs_all[name].append(dets_outputs['feature_vector'])

            current_sample_token = nusc.get('sample', current_sample_token)['next']

            current_ground_truths = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

        processed_scene_tokens.add(scene_token)

    
    svd_matrices = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
    mean_vectors = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
    std_vectors = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
    
    # Perform SVD for each tracking name
    for name in NUSCENES_TRACKING_NAMES:
        if len(pcds_all[name]) == 0:
            continue
        print(len(pcds_all[name]), name)
        # Stack the features (shape: (num_samples, 512, 3, 3))
        pcds_all[name] = np.vstack(pcds_all[name])

        # Initialize dictionaries to store SVD results per grid position
        svd_matrices[name] = {}  # Store projection matrices per grid position
        mean_vectors[name] = {}
        std_vectors[name] = {}

        # Loop through each grid position in the 3x3 grid
        for i in range(3):
            for j in range(3):
                # Extract features for this grid position (shape: (num_samples, 512))
                grid_features = pcds_all[name][:, :, i, j]

                # Compute mean and std for normalization
                mean_vector = np.mean(grid_features, axis=0)  # Shape: (512,)
                std_vector = np.std(grid_features, axis=0)    # Shape: (512,)
                std_vector[std_vector == 0] = 1e-8  # Avoid division by zero

                # Mean shift and normalize
                normalized_features = (grid_features - mean_vector) / std_vector  # Shape: (num_samples, 512)

                # Convert to tensor for SVD
                normalized_tensor = torch.tensor(normalized_features, dtype=torch.float32)

                # Run SVD
                print(f"Running SVD for {name}, grid position ({i}, {j})...")
                U, S, Vh = torch.linalg.svd(normalized_tensor, full_matrices=False)

                # Save the results for this grid position
                svd_matrices[name][(i, j)] = Vh.cpu().numpy()  # Projection matrix (shape: 512x512)
                mean_vectors[name][(i, j)] = mean_vector
                std_vectors[name][(i, j)] = std_vector

        print(f"Completed SVD for {name}.")

    # Save the SVD matrices, mean vectors, and std vectors
    with open(args.output_path, 'wb') as f:
        pickle.dump({
            'svd_matrices': svd_matrices,
            'mean_vectors': mean_vectors,
            'std_vectors': std_vectors
        }, f)

    print(f"Saved SVD projection matrices to {args.output_path}.")

if __name__ == "__main__":
    svd()