from __future__ import print_function
from tqdm import tqdm
import pickle
import argparse
from nuscenes import NuScenes
import numpy as np
import torch
import umap
from tqdm import tqdm
import os.path
from itertools import product
import pickle
import argparse
from nuscenes import NuScenes
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import umap
import numpy as np
import multiprocessing as mp
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE, Isomap
from scipy.sparse import lil_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    parser.add_argument('--output_path', type=str, default="svd_matrices_spatial_51233.pkl", help='Path to save the SVD projection matrices')

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

    
    os.makedirs('svd_umap/umap3d', exist_ok=True)
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine')
    # Perform SVD for each tracking name
    for name in NUSCENES_TRACKING_NAMES:
        if len(pcds_all[name]) == 0:
            continue
        print(len(pcds_all[name]), name)
        # Stack the features (shape: (num_samples, 512, 3, 3))
        pcds_all[name] = np.vstack(pcds_all[name])

        top_features = 64
        projected_features = np.zeros((len(pcds_all[name]), top_features, 3, 3))

        for i, j in product(range(3), range(3)):
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

            # Project using SVD
            reduced_projection_matrix = Vh[:top_features, :]
            projected_grid_features = np.dot(normalized_features, reduced_projection_matrix.T)

            projected_features[:, :, i, j] = projected_grid_features


        pcds = projected_features.reshape(len(pcds_all[name]), -1)
        pcds_umap = reducer.fit_transform(pcds)
        print(pcds.shape, 'umap')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcds_umap[:, 0], pcds_umap[:, 1], pcds_umap[:, 2], alpha=0.6)
        ax.set_title(f'3D UMAP of pcds for {name}')
        plt.savefig(f'svd_umap/umap3d/umap_3d_pcds_{name}.png')
        plt.close()


if __name__ == "__main__":
    svd()